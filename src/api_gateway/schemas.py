# src/api_gateway/schemas.py
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Тело запроса к /api/v1/chat/stream."""

    query: str = Field(
        ...,
        description="Текст запроса от пользователя",
        min_length=1,
        max_length=1500,
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
