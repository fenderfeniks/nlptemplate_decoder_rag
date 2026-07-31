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

    # 1. Ретривал выполняется ДО начала стриминга.
    # Если RAG упадет, FastAPI корректно перехватит HTTPException и вернет 502 JSON.
    prompt = await orchestrator.build_prompt(
        query=body.query,
        top_k=body.top_k,
        filters=body.filters,
    )

    # 2. Генератор занимается ИСКЛЮЧИТЕЛЬНО стримингом LLM
    async def _stream_generator() -> AsyncIterator[str]:
        try:
            async for chunk in orchestrator.llm_client.generate_stream(prompt):
                yield chunk
        except Exception as e:
            # Ошибки самой генерации (когда заголовки 200 уже ушли) глушим маркером
            logger.exception("Непредвиденная ошибка при генерации ответа: %s", e)
            yield "\n[Ошибка при получении ответа]"

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")
