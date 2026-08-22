# src/api_gateway/endpoints/chat.py
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from src.api_gateway.dependencies import get_orchestrator
from src.api_gateway.metrics import LLM_REQUESTS_IN_FLIGHT, LLM_TRUNCATED_RESPONSES_TOTAL
from src.api_gateway.metrics_recorder import (
    classify_error,
    record_error,
    record_stream_metrics,
    record_ttft,
)
from src.api_gateway.rag_logger import log_rag_triple
from src.api_gateway.resilience import CircuitBreakerError
from src.api_gateway.schemas import ChatRequest
from src.application.orchestrator import RAGOrchestrator
from src.pipelines.decoder.inference.response_cleaner import ResponseCleaner


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])
tracer = trace.get_tracer(__name__)

_stream_cleaner = ResponseCleaner.for_stream()
_batch_cleaner = ResponseCleaner.for_batch()


@router.post("/stream", summary="Стриминг ответа RAG-системы")
async def chat_stream_endpoint(
    body: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    history_dicts = [msg.model_dump() for msg in body.chat_history] if body.chat_history else None

    # build_prompt теперь возвращает BuildPromptResult с docs и флагом деградации.
    # Документы нужны здесь для rag_logger — без повторного запроса к RAG.
    result = await orchestrator.build_prompt(
        query=body.query,
        chat_history=history_dicts,
        top_k=body.top_k,
        filters=body.filters,
    )

    # Заголовок X-RAG-Status сигнализирует клиенту о деградации.
    # Клиент может показать: "Ответ без учёта документов (RAG недоступен)".
    response_headers = {
        "X-Request-Id": request_id,
        "X-RAG-Status": "degraded" if result.rag_degraded else "ok",
    }

    LLM_REQUESTS_IN_FLIGHT.inc()
    wall_start = time.perf_counter()

    async def _stream_generator() -> AsyncIterator[str]:
        ttft_s = 0.0
        ttft_recorded = False
        status = "success"
        model_name = orchestrator.llm_client.model_name
        generated_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async for chunk in orchestrator.llm_client.generate_stream(
                result.prompt,
                request_id=request_id,  # пробрасываем в OTEL span
            ):
                if chunk.is_final:
                    prompt_tokens = chunk.prompt_tokens
                    completion_tokens = chunk.completion_tokens
                    continue

                clean_piece = _stream_cleaner.clean(chunk.text)
                if not clean_piece:
                    continue

                if not ttft_recorded:
                    ttft_s = time.perf_counter() - wall_start
                    record_ttft(model=model_name, ttft_s=ttft_s)
                    ttft_recorded = True

                generated_text += clean_piece
                yield clean_piece

        except CircuitBreakerError:
            # LLM circuit breaker открыт — сервис временно недоступен.
            # 503 семантически точнее 502: мы сами отказываем, не downstream.
            status = "circuit_breaker_open"
            record_error(error_type="circuit_breaker_open")
            logger.warning("[%s] LLM circuit breaker открыт, fast fail", request_id)
            yield "\n[Сервис временно недоступен. Попробуйте позже.]"

        except Exception as e:
            status = "error"
            record_error(error_type=classify_error(e))
            logger.exception("[%s] ошибка стриминга: %s", request_id, e)
            yield "\n[Ошибка при получении ответа]"

        finally:
            elapsed = time.perf_counter() - wall_start
            cleaned_for_metrics = _batch_cleaner.clean(generated_text)

            if not completion_tokens and cleaned_for_metrics:
                completion_tokens = len(cleaned_for_metrics.split())
                logger.debug(
                    "[%s] usage не получен от сервера, completion_tokens аппроксимирован: %d",
                    request_id,
                    completion_tokens,
                )

            if cleaned_for_metrics and cleaned_for_metrics[-1] not in ".!?":
                LLM_TRUNCATED_RESPONSES_TOTAL.labels(model=model_name).inc()
                logger.warning(
                    "[%s] ответ выглядит усечённым (не заканчивается на . ! ?)",
                    request_id,
                )

            record_stream_metrics(
                request_id=request_id,
                model=model_name,
                status=status,
                elapsed=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_s=ttft_s,
                generated_text=cleaned_for_metrics,
            )

            # Получаем trace_id из активного OTEL span'а для кросс-ссылки
            # между JSONL-логом в Kibana и трейсом в Jaeger.
            current_span = trace.get_current_span()
            otel_trace_id: str | None = None
            if current_span and current_span.is_recording():
                otel_trace_id = format(current_span.get_span_context().trace_id, "032x")

            # Логируем тройку только при успешной генерации (или деградации RAG).
            # При circuit_breaker_open ответа нет — логировать нечего.
            if status in ("success",) or result.rag_degraded:
                log_rag_triple(
                    request_id=request_id,
                    query=body.query,
                    retrieved_docs=result.retrieved_docs,
                    response=cleaned_for_metrics,
                    model=model_name,
                    rag_degraded=result.rag_degraded,
                    elapsed_s=elapsed,
                    trace_id=otel_trace_id,
                )

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers=response_headers,
    )
