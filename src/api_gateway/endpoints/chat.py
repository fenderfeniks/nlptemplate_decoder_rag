# src/api_gateway/endpoints/chat.py
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api_gateway.dependencies import get_orchestrator
from src.api_gateway.metrics import LLM_REQUESTS_IN_FLIGHT
from src.api_gateway.metrics_recorder import (
    classify_error,
    record_error,
    record_stream_metrics,
    record_ttft,
)
from src.api_gateway.schemas import ChatRequest
from src.application.orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/stream", summary="Стриминг ответа RAG-системы")
async def chat_stream_endpoint(
    body: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    history_dicts = [msg.model_dump() for msg in body.chat_history] if body.chat_history else None

    prompt = await orchestrator.build_prompt(
        query=body.query,
        chat_history=history_dicts,
        top_k=body.top_k,
        filters=body.filters,
    )

    LLM_REQUESTS_IN_FLIGHT.inc()
    wall_start = time.perf_counter()

    async def _stream_generator() -> AsyncIterator[str]:
        ttft_s = 0.0
        ttft_recorded = False
        status = "success"
        model_name = orchestrator.llm_client.model_name
        generated_text = ""

        try:
            async for text_piece in orchestrator.llm_client.generate_stream(prompt):
                if not ttft_recorded and text_piece:
                    ttft_s = time.perf_counter() - wall_start
                    record_ttft(model=model_name, ttft_s=ttft_s)
                    ttft_recorded = True

                generated_text += text_piece
                yield text_piece

        except Exception as e:
            status = "error"
            record_error(error_type=classify_error(e))
            logger.exception("[%s] ошибка стриминга: %s", request_id, e)
            yield "\n[Ошибка при получении ответа]"
        finally:
            elapsed = time.perf_counter() - wall_start
            record_stream_metrics(
                request_id=request_id,
                model=model_name,
                status=status,
                elapsed=elapsed,
                prompt_tokens=0,  # llama.cpp не отдаёт токены в стриме без stream_options
                completion_tokens=len(generated_text.split()),  # аппроксимация
                ttft_s=ttft_s,
                generated_text=generated_text,
            )

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")