import pytest
from pydantic import ValidationError

from src.pipelines.rag.core.data.schemas import RAGIndexingRecord, RAGTrainingRecord


class TestRAGIndexingRecord:
    def test_valid_indexing_record(self):
        """Успешное создание записи для индексации."""
        record = RAGIndexingRecord(text="Длинный текст для векторной базы.", metadata={"id": 1})
        assert record.text == "Длинный текст для векторной базы."
        assert record.metadata == {"id": 1}

    def test_text_too_short(self):
        """Ошибка, если текст короче 10 символов."""
        with pytest.raises(ValidationError, match="Текст слишком короткий"):
            RAGIndexingRecord(text="Коротко")

    def test_empty_text(self):
        """Ошибка, если текст пустой."""
        with pytest.raises(ValidationError, match="не может быть пустым"):
            RAGIndexingRecord(text="   ")


class TestRAGTrainingRecord:
    def test_valid_contrastive_record(self):
        """Успешное создание записи для Contrastive Learning (без negative)."""
        record = RAGTrainingRecord(query="Запрос", positive_doc="Позитивный док")
        assert record.mode == "contrastive"
        assert record.negative_doc is None

    def test_valid_triplet_record(self):
        """Успешное создание записи для Triplet Loss (с negative)."""
        record = RAGTrainingRecord(
            query="Запрос", 
            positive_doc="Позитивный док", 
            negative_doc="Негативный док"
        )
        assert record.mode == "triplet"
        assert record.negative_doc == "Негативный док"

    def test_negative_equals_positive(self):
        """Ошибка, если негативный документ совпадает с позитивным."""
        with pytest.raises(ValidationError, match="не должен совпадать с positive_doc"):
            RAGTrainingRecord(query="Запрос", positive_doc="Одинаково", negative_doc="Одинаково")

    def test_empty_fields(self):
        """Ошибка при пустых обязательных полях."""
        with pytest.raises(ValidationError, match="слишком короткий"):
            RAGTrainingRecord(query="Запрос", positive_doc="Да")
            
        with pytest.raises(ValidationError, match="является пустой строкой"):
            RAGTrainingRecord(query="Запрос", positive_doc="Позитив", negative_doc="   ")