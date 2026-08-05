# src/api_gateway/endpoints/chat.py
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api_gateway.dependencies import get_orchestrator
from src.api_gateway.schemas import ChatRequest
from src.application.orchestrator import RAGOrchestrator


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/stream", summary="Стриминг ответа RAG-системы")
async def chat_stream_endpoint(
    body: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    """Принимает вопрос пользователя, ретривает документы и стримит ответ LLM."""

    # Конвертируем Pydantic модели Message обратно в словари для оркестратора
    history_dicts = [msg.model_dump() for msg in body.chat_history] if body.chat_history else None

    prompt = await orchestrator.build_prompt(
        query=body.query,
        chat_history=history_dicts,
        top_k=body.top_k,
        filters=body.filters,
    )

    async def _stream_generator() -> AsyncIterator[str]:
        try:
            async for chunk in orchestrator.llm_client.generate_stream(prompt):
                yield chunk
        except Exception as e:
            logger.exception("Непредвиденная ошибка при генерации ответа: %s", e)
            yield "\n[Ошибка при получении ответа]"

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")
