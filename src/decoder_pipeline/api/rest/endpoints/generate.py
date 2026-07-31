import asyncio
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.decoder_pipeline.api.metrics import LLM_GENERATED_TOKENS_TOTAL, LLM_GENERATION_TIME
from src.decoder_pipeline.api.rest.dependencies import get_generator, get_prompt_manager
from src.decoder_pipeline.api.rest.limiter import limiter
from src.decoder_pipeline.api.schemas import GenerationRequest, GenerationResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Generation"])


@router.post("/generate/stream")
@limiter.limit("5/minute")
async def generate_stream_endpoint(request: Request, body: GenerationRequest) -> StreamingResponse:
    generator = get_generator(request)
    prompt_manager = get_prompt_manager(request)
    cfg = request.app.state.config

    api_gen_cfg = cfg.api.get("generation", {})
    template_name = api_gen_cfg.get("default_template", "rag_qa")
    static_context = api_gen_cfg.get("static_context", "")

    final_prompt = prompt_manager.render(
        template_name=template_name,
        question=body.prompt,
        context=static_context,
    )

    async def _async_stream() -> AsyncIterator[str]:
        try:
            # Больше никаких семафоров!
            async for chunk in generator.generate_stream(final_prompt):
                yield chunk
        except asyncio.CancelledError:
            logger.warning("Клиент разорвал соединение при стриминге!")
            raise

    return StreamingResponse(_async_stream(), media_type="text/event-stream")


@router.post("/generate", response_model=GenerationResponse)
@limiter.limit("5/minute")
async def generate_text(
    request: Request,
    body: GenerationRequest,
) -> GenerationResponse:
    generator = get_generator(request)
    prompt_manager = get_prompt_manager(request)
    cfg = request.app.state.config

    api_gen_cfg = cfg.api.get("generation", {})
    template_name = api_gen_cfg.get("default_template", "rag_qa")
    static_context = api_gen_cfg.get("static_context", "")

    final_prompt = prompt_manager.render(
        template_name=template_name,
        question=body.prompt,
        context=static_context,
    )

    try:
        start_time = time.perf_counter()

        with LLM_GENERATION_TIME.labels(source="rest").time():
            # Напрямую вызываем асинхронный метод клиента, убрав asyncio.to_thread
            results = await generator(final_prompt)

        elapsed_time = time.perf_counter() - start_time
        generated_text = results[0]["generated_text"]

        approx_tokens = len(generated_text.split()) * 1.3
        LLM_GENERATED_TOKENS_TOTAL.labels(source="rest").inc(approx_tokens)

        tps = approx_tokens / elapsed_time if elapsed_time > 0 else 0
        logger.info("Генерация завершена | %.0f токенов | %.2f t/s", approx_tokens, tps)

        return GenerationResponse(generated_text=generated_text)

    except asyncio.CancelledError:
        logger.warning("Клиент разорвал соединение! Генерация прервана.")
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ошибка инференса: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка связи с LLM-сервером.") from e
