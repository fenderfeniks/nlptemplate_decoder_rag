import pytest
from datasets import Dataset

from src.pipelines.rag.core.data.transforms.validation import RAGValidationTransform


class TestRAGValidationTransform:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Неизвестный режим валидации"):
            RAGValidationTransform(mode="invalid")

    def test_indexing_validation_methods_direct(self):
        """Прямое тестирование логики валидации индексации."""
        transform = RAGValidationTransform(mode="indexing")
        
        assert transform._get_required_columns() == ["text"]
        assert transform._get_filter_column() == "text"
        
        batch = {
            "text": ["Валидный длинный текст", "Мало", None],
            "metadata": [{"id": 1}, {"id": 2}, None]
        }
        res = transform._validate_batch(batch)
        
        # Битая запись "Мало" и None должны замениться на ""
        assert res["text"] == ["Валидный длинный текст", "", ""]
        assert res["metadata"] == [{"id": 1}, {}, {}]

    def test_contrastive_validation_methods_direct(self):
        """Прямое тестирование логики contrastive-валидации."""
        transform = RAGValidationTransform(mode="contrastive")
        
        assert transform._get_required_columns() == ["query", "positive_doc"]
        assert transform._get_filter_column() == "query"
        
        batch = {
            "query": ["Запрос 1", "З", "Запрос 3"],
            "positive_doc": ["Хороший позитив", "Хороший позитив 2", "Хороший позитив 3"],
            "negative_doc": ["Негатив 1", "Негатив 2", "Хороший позитив 3"] # Совпадает с позитивом (ошибка)
        }
        res = transform._validate_batch(batch)
        
        # Индекс 1 (короткий query) и индекс 2 (neg == pos) отбрасываются
        assert res["query"] == ["Запрос 1", "", ""]
        assert res["positive_doc"] == ["Хороший позитив", "", ""]
        assert res["negative_doc"] == ["Негатив 1", None, None]

    def test_full_pipeline_indexing(self):
        """Интеграционный тест: __call__ отфильтровывает пустые значения."""
        ds = Dataset.from_dict({"text": ["Нормальный текст для RAG", "Мало"]})
        transform = RAGValidationTransform(mode="indexing", num_proc=1)
        
        result = transform(ds)
        assert len(result) == 1
        assert result["text"] == ["Нормальный текст для RAG"]