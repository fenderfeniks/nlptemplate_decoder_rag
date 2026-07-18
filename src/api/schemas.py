# src/api/schemas.py

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Роль: 'user' или 'assistant'")
    content: str = Field(..., description="Текст сообщения")


class ChatRequest(BaseModel):
    query: str = Field(..., description="Текущий запрос от пользователя")
    # ДОБАВИЛИ ПАМЯТЬ: Список предыдущих сообщений
    history: list[ChatMessage] | None = Field(default=[], description="История диалога")
    use_rag: bool = Field(default=True)
    max_tokens: int | None = Field(default=256)


class ChatResponse(BaseModel):
    answer: str
    context_used: str | None = None
