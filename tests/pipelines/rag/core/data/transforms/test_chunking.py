# tests/pipelines/rag/core/data/transforms/test_chunking.py
import pytest
from datasets import Dataset

from src.pipelines.rag.core.data.transforms.chunking import OverlappingChunkingTransform


class TestOverlappingChunkingTransform:
    def test_invalid_init(self):
        with pytest.raises(ValueError, match="должен быть положительным"):
            OverlappingChunkingTransform(chunk_size=-1)
        with pytest.raises(ValueError, match="должен быть строго меньше chunk_size"):
            OverlappingChunkingTransform(chunk_size=10, chunk_overlap=15)

    def test_missing_column_skipped(self):
        ds = Dataset.from_dict({"wrong_col": ["text"]})
        transform = OverlappingChunkingTransform(num_proc=None)
        assert transform(ds) is ds

    def test_chunking_logic(self):
        """Проверка нарезки с перекрытием с точным математическим расчетом."""
        ds = Dataset.from_dict({
            "text": ["Один два три четыре пять шесть"],
            "meta": ["doc1"]
        })
        
        # num_proc=None заставляет HF выполнять map в главном потоке (сохраняя покрытие)
        transform = OverlappingChunkingTransform(chunk_size=15, chunk_overlap=6, num_proc=None)
        result = transform(ds)
        
        chunks = result["text"]
        # Корректная разбивка:
        # 1. "Один два три" (12 симв)
        # 2. "три четыре" (10 симв, "четыре" в перекрытие не влезло, так как 7 > 6)
        # 3. "пять шесть" (10 симв)
        assert chunks == ["Один два три", "три четыре", "пять шесть"]
        assert result["meta"] == ["doc1", "doc1", "doc1"]

    def test_empty_and_none_text(self):
        """Пустые тексты и None должны возвращать один пустой чанк."""
        ds = Dataset.from_dict({"text": ["", None]})
        transform = OverlappingChunkingTransform(num_proc=None)
        result = transform(ds)
        
        assert len(result) == 2
        assert result["text"] == ["", ""]