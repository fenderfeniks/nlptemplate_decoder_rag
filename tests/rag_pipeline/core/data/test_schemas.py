# tests/rag_pipeline/core/data/test_schemas.py
import pytest
from pydantic import ValidationError

from src.rag_pipeline.core.data.schemas import RAGIndexingRecord, RAGTrainingRecord


class TestRAGIndexingRecord:
    def test_valid_record_accepted(self):
        record = RAGIndexingRecord(text="Это валидный текст длиной больше 10 символов.", metadata={"id": 1})
        assert record.text == "Это валидный текст длиной больше 10 символов."
        assert record.metadata == {"id": 1}

    def test_short_text_rejected(self):
        with pytest.raises(ValidationError) as exc:
            RAGIndexingRecord(text="Короткий")
        # Pydantic v2 проверяет длину поля до пользовательского валидатора
        assert "String should have at least 10 characters" in str(exc.value)

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            RAGIndexingRecord(text="   ")

    def test_default_metadata_is_empty_dict(self):
        record = RAGIndexingRecord(text="Длинный текст без метаданных")
        assert record.metadata == {}


class TestRAGTrainingRecord:
    def test_valid_triplet_accepted(self):
        record = RAGTrainingRecord(
            query="Как работает RAG?",
            positive_doc="RAG работает через поиск...",
            negative_doc="Рецепт борща..."
        )
        assert record.query == "Как работает RAG?"
        assert record.negative_doc == "Рецепт борща..."

    def test_missing_negative_is_allowed(self):
        """Негативный документ опционален (default=None)."""
        record = RAGTrainingRecord(
            query="Запрос",
            positive_doc="Позитивный ответ"
        )
        assert record.negative_doc is None

    def test_empty_query_or_positive_rejected(self):
        with pytest.raises(ValidationError):
            RAGTrainingRecord(query="   ", positive_doc="Doc")
        
        with pytest.raises(ValidationError):
            RAGTrainingRecord(query="Query", positive_doc="")