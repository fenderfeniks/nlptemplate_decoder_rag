# src/rag_pipeline/core/data/schemas.py
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RAGIndexingRecord(BaseModel):
    """Схема для сырых данных при подготовке векторной базы (Индексация)."""
    
    text: str = Field(description="Сырой текст документа/статьи")
    metadata: Optional[dict] = Field(default_factory=dict, description="Метаданные (URL, title, date)")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Текст документа не может быть пустым")
        if len(v) < 10:
            raise ValueError("Текст слишком короткий для индексации (минимум 10 символов)")
        return v


class RAGTrainingRecord(BaseModel):
    """Схема для обучения энкодера (Contrastive Learning / Triplet Loss)."""
    
    query: str = Field(description="Поисковый запрос")
    positive_doc: str = Field(description="Релевантный документ")
    negative_doc: Optional[str] = Field(default=None, description="Нерелевантный документ (Hard Negative)")

    @field_validator("query", "positive_doc", "negative_doc")
    @classmethod
    def validate_content_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Поле не может быть пустым")
        return v