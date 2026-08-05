from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="Роль: 'user' или 'assistant'")
    content: str = Field(..., description="Текст сообщения")


class ChatRequest(BaseModel):
    """Тело запроса к /api/v1/chat/stream."""

    query: str = Field(
        ...,
        description="Текст запроса от пользователя",
        min_length=1,
        max_length=1500,
    )
    chat_history: list[Message] | None = Field(
        default=None,
        description="История предыдущих сообщений для контекста диалога",
    )
    top_k: int | None = Field(
        None,
        description="Количество документов для ретривала",
        ge=1,
        le=10,
    )
    filters: dict | None = Field(
        None,
        description="Опциональные фильтры для векторной БД",
    )
