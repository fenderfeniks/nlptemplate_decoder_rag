# src/pipelines/rag/api/schemas.py
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Текст запроса для поиска",
        min_length=2,
        max_length=1000,
    )
    top_k: int = Field(
        5,
        description="Количество возвращаемых документов",
        ge=1,
        le=50,
    )
    filters: dict[str, Any] | None = Field(
        None,
        description="Фильтры по метаданным для точного совпадения (опционально)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Как работает HNSW?",
                "top_k": 3,
                "filters": {"language": "ru"},
            }
        }
    )


class Document(BaseModel):
    """Один найденный документ с косинусным score и метаданными."""

    score: float = Field(
        ...,
        description="Косинусное сходство запроса и документа. "
        "FAISS IndexFlatIP с нормализованными векторами возвращает inner product "
        "в [-1, 1]; значения вне этого диапазона клипаются во избежание ошибок.",
    )
    metadata: dict[str, Any] = Field(
        description="Метаданные документа: text, doc_id, url, title и т.д."
    )

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Клипает score в [-1.0, 1.0].

        FAISS с нормализованными векторами возвращает inner product в [-1, 1],
        но floating point неточности могут дать 1.0000001 — это вызовет
        ValidationError. Клипаем явно вместо надежды на точность.
        """
        return max(-1.0, min(1.0, float(v)))


class SearchResponse(BaseModel):
    """Ответ на поисковый запрос."""

    results: list[Document] = Field(description="Список найденных документов по убыванию score")
    total: int = Field(description="Количество возвращённых документов")
    query_time_ms: float = Field(description="Время поиска в миллисекундах")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "score": 0.923,
                        "metadata": {
                            "doc_id": "abc123",
                            "text": "HNSW — иерархический граф для приближённого поиска...",
                            "url": "https://example.com/hnsw",
                        },
                    }
                ],
                "total": 1,
                "query_time_ms": 12.4,
            }
        }
    )
