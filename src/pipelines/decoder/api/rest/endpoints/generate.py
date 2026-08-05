import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.pipelines.decoder.api.metrics import (
    LLM_GENERATED_TOKENS_TOTAL,
    LLM_GENERATION_REQUESTS_TOTAL,
    LLM_GENERATION_TIME,
)
from src.pipelines.decoder.api.rest.dependencies import get_generator, get_prompt_manager
from src.pipelines.decoder.api.rest.limiter import limiter
from src.pipelines.decoder.api.schemas import GenerationRequest, GenerationResponse
from src.pipelines.decoder.core.prompts.manager import PromptManager
from src.pipelines.decoder.inference.inference import LLMGenerationClient


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Generation"])


def _build_prompt(request: Request, body: GenerationRequest, prompt_manager: PromptManager) -> str:
    """Строит финальный промпт из конфига и тела запроса.

    Вынесено в отдельную функцию, чтобы не дублировать между stream и non-stream эндпоинтами.
    """
    cfg = request.app.state.config
    api_gen_cfg = cfg.decoder_pipeline.get("generation", {})
    template_name = api_gen_cfg.get("default_template", "rag_qa")
    static_context = api_gen_cfg.get("static_context", "")

    return prompt_manager.render(
        template_name=template_name,
        question=body.prompt,
        context=static_context,
    )


@router.post("/generate/stream")
@limiter.limit("5/minute")
async def generate_stream_endpoint(
    request: Request,
    body: GenerationRequest,
    generator: LLMGenerationClient = Depends(get_generator),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
) -> StreamingResponse:
    """Стриминг ответа LLM токен за токеном."""
    LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest_stream", status="started").inc()
    final_prompt = _build_prompt(request, body, prompt_manager)

    async def _async_stream():
        try:
            async for chunk in generator.generate_stream(final_prompt):
                yield chunk
            LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest_stream", status="success").inc()
        except asyncio.CancelledError:
            logger.warning("Клиент разорвал соединение при стриминге!")
            LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest_stream", status="cancelled").inc()
            raise

    return StreamingResponse(_async_stream(), media_type="text/event-stream")


@router.post("/generate", response_model=GenerationResponse)
@limiter.limit("5/minute")
async def generate_text(
    request: Request,
    body: GenerationRequest,
    generator: LLMGenerationClient = Depends(get_generator),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
) -> GenerationResponse:
    """Синхронная генерация с полным ответом."""
    LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest", status="started").inc()
    final_prompt = _build_prompt(request, body, prompt_manager)

    try:
        with LLM_GENERATION_TIME.labels(source="rest").time():
            # Вызываем правильный метод
            results = await generator.generate(final_prompt)

        # Берем строку напрямую
        generated_text = results[0]

        # Аппроксимация токенов через split — достаточно для метрики нагрузки.
        # Точный подсчёт требует токенизатора или response.usage от vLLM.
        approx_tokens = len(generated_text.split()) * 1.3
        LLM_GENERATED_TOKENS_TOTAL.labels(source="rest").inc(approx_tokens)
        LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest", status="success").inc()

        logger.info("Генерация завершена | ~%.0f токенов", approx_tokens)
        return GenerationResponse(generated_text=generated_text)

    except asyncio.CancelledError:
        logger.warning("Клиент разорвал соединение! Генерация прервана.")
        LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest", status="cancelled").inc()
        raise
    except HTTPException:
        LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest", status="error").inc()
        raise
    except Exception as e:
        logger.exception("Ошибка инференса: %s", e)
        LLM_GENERATION_REQUESTS_TOTAL.labels(source="rest", status="error").inc()
        raise HTTPException(status_code=500, detail="Ошибка связи с LLM-сервером.") from e
