import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=10)
    filters: dict | None = None


@router.post("/stream")
async def chat_stream_endpoint(request: Request, body: ChatRequest) -> StreamingResponse:
    """Стриминг ответа RAG-системы."""
    orchestrator = request.app.state.orchestrator

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Оркестратор не готов.")

    async def _stream_generator() -> AsyncIterator[str]:
        try:
            # Делегируем всю магию оркестратору
            async for chunk in orchestrator.ask_stream(
                query=body.query, top_k=body.top_k, filters=body.filters
            ):
                yield chunk
        except Exception as e:
            logger.error("Ошибка при генерации ответа: %s", e)
            # В SSE можно передать маркер ошибки клиенту, если нужно
            yield "\n[Ошибка при получении ответа]"

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")
