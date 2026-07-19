import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.metrics import LLM_GENERATIONS_TOTAL, LLM_INFERENCE_TIME
from src.api.rest.dependencies import get_generator, get_prompt_manager
from src.api.rest.limiter import limiter
from src.api.schemas import ChatRequest, ChatResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Generation"])


@router.post("/generate", response_model=ChatResponse)
@limiter.limit("5/minute")
async def generate_text(
    request: Request,
    body: ChatRequest,
    generator=Depends(get_generator),
    prompt_manager=Depends(get_prompt_manager),
):
    try:
        context = None

        history_text = ""
        if body.history:
            history_text = "История предыдущего диалога:\n"
            for msg in body.history:
                history_text += f"{msg.role.capitalize()}: {msg.content}\n"
            history_text += "\n"

        full_query = history_text + body.query

        # Ленивая загрузка ретривера
        if body.use_rag:
            retriever = request.app.state.ml_models.get("retriever")
            if not retriever:
                raise HTTPException(status_code=503, detail="RAG is not configured or unavailable")

            context = await asyncio.to_thread(retriever.retrieve_context, body.query)

            final_prompt = prompt_manager.build_rag_prompt(
                query=full_query,
                context=context,
            )
        else:
            final_prompt = prompt_manager.build_simple_prompt(full_query)

        local_gen_kwargs = {}
        if body.max_tokens:
            local_gen_kwargs["max_new_tokens"] = body.max_tokens

        LLM_GENERATIONS_TOTAL.labels(source="rest").inc()

        with LLM_INFERENCE_TIME.labels(source="rest").time():
            responses = await asyncio.to_thread(
                generator.generate, final_prompt, **local_gen_kwargs
            )

        return ChatResponse(answer=responses[0], context_used=context)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка инференса: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка генерации ответа.") from e
