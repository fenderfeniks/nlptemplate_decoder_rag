# src/api_gateway/endpoints/chat.py
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
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

    async def _stream_generator() -> AsyncIterator[str]:
        try:
            async for chunk in orchestrator.ask_stream(
                query=body.query,
                top_k=body.top_k,
                filters=body.filters,
            ):
                yield chunk
        except HTTPException:
            # Пробрасываем HTTP-ошибки (например, 502 от RAG API) как есть —
            # они уже несут корректный статус-код, не нужно их поглощать.
            raise
        except Exception as e:
            # Непредвиденные ошибки: логируем и сигнализируем клиенту
            # маркером внутри SSE-потока (HTTP-заголовки уже отправлены).
            logger.exception("Непредвиденная ошибка при генерации ответа: %s", e)
            yield "\n[Ошибка при получении ответа]"

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")
