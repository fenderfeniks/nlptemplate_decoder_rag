# src/api/schemas.py
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Роль: 'user' или 'assistant'", max_length=50)
    content: str = Field(..., description="Текст сообщения", max_length=2000)


class ChatRequest(BaseModel):
    query: str = Field(..., description="Текущий запрос от пользователя", max_length=2000)
    history: list[ChatMessage] | None = Field(
        default=[], description="История диалога", max_length=10
    )
    use_rag: bool = Field(default=True)
    max_tokens: int | None = Field(default=256, le=1024)


class ChatResponse(BaseModel):
    answer: str
    context_used: str | None = None
