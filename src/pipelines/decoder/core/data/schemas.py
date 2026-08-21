# src/pipelines/decoder/core/data/schemas.py
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class RawDatasetRecord(BaseModel):
    """Контракт для сырой записи датасета перед токенизацией (SFT и RAG)."""

    prompt: str = Field(description="Входной промпт для модели (или вопрос)")
    target: str = Field(description="Ожидаемый ответ")
    
    # --- НОВЫЕ ОПЦИОНАЛЬНЫЕ ПОЛЯ ---
    context: Optional[str] = Field(default=None, description="Извлеченный контекст (для RAG)")
    system_prompt: Optional[str] = Field(default=None, description="Опциональный системный промпт")

    @field_validator("prompt")
    @classmethod
    def validate_prompt_length(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Промпт не может быть пустым")
        if len(v) < 3:
            raise ValueError("Промпт слишком короткий (минимум 3 символа)")
        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Target передан, но является пустой строкой")
        return v