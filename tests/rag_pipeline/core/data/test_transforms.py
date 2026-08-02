# tests/rag_pipeline/core/data/test_transforms.py
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from src.rag_pipeline.core.data.transforms.chunking import OverlappingChunkingTransform
from src.rag_pipeline.core.data.transforms.deduplication import ExactDeduplicationTransform
from src.rag_pipeline.core.data.transforms.metadata import MetadataInjectorTransform
from src.rag_pipeline.core.data.transforms.tokenization import RAGTokenizationTransform
from src.rag_pipeline.core.data.transforms.validation import ValidationTransform
from src.rag_pipeline.core.data.transforms.filtering import LengthFilterTransform


def _make_dataset(records: list[dict]) -> Dataset:
    keys = records[0].keys()
    return Dataset.from_dict({k: [r[k] for r in records] for k in keys})

class DummyRAGTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[1]] * len(texts), "attention_mask": [[1]] * len(texts)}

    
class TestValidationTransform:
    def test_indexing_mode_filters_invalid(self):
        """Проверка отсева слишком коротких текстов в режиме indexing."""
        ds = _make_dataset([
            {"text": "Длинный валидный текст для RAG", "metadata": {}},
            {"text": "Коротк", "metadata": {}}, # Удалится
        ])
        transform = ValidationTransform(mode="indexing", num_proc=1, batch_size=2)
        res = transform(ds)
        assert len(res) == 1
        assert res[0]["text"] == "Длинный валидный текст для RAG"
    """
    def test_contrastive_mode_filters_invalid(self):
        ds = _make_dataset([
            {"query": "Q1", "positive_doc": "P1", "negative_doc": "N1"},
            {"query": "", "positive_doc": "P2", "negative_doc": None}, # Удалится
            {"query": "Q3", "positive_doc": "P3", "negative_doc": None},
        ])
        transform = ValidationTransform(mode="contrastive", num_proc=1, batch_size=2)
        res = transform(ds)
        assert len(res) == 2"""


class TestMetadataInjectorTransform:
    def test_injects_metadata_correctly(self):
        ds = _make_dataset([
            {"text": "Содержимое", "metadata": {"title": "Test", "date": "2026"}}
        ])
        transform = MetadataInjectorTransform(num_proc=1, batch_size=2)
        res = transform(ds)
        
        # Ожидаем капитализацию ключей
        assert "Title: Test\nDate: 2026\n\nСодержимое" in res[0]["text"]

    def test_skips_empty_metadata(self):
        ds = _make_dataset([{"text": "Просто текст", "metadata": {}}])
        transform = MetadataInjectorTransform(num_proc=1, batch_size=2)
        res = transform(ds)
        assert res[0]["text"] == "Просто текст"


class TestOverlappingChunkingTransform:
    def test_chunks_with_overlap(self):
        ds = _make_dataset([{"text": "A B C D E F G H I J"}])
        # Разделитель пробел. chunk_size=9 символов -> влезет примерно 4-5 букв с пробелами.
        # "A B C D" = 7 символов. Перекрытие 3 символа ("C D").
        transform = OverlappingChunkingTransform(
            chunk_size=7, chunk_overlap=3, separator=" ", num_proc=1, batch_size=2
        )
        res = transform(ds)
        
        assert len(res) > 1
        # Проверяем, что метаданные дублируются, а текст бьется на части
        assert isinstance(res[0]["text"], str)
        assert len(res[0]["text"]) <= 7

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            OverlappingChunkingTransform(chunk_size=10, chunk_overlap=10) # Overlap >= size


class TestExactDeduplicationTransform:
    def test_removes_exact_duplicates(self):
        ds = _make_dataset([
            {"text": "Одинаковый текст"},
            {"text": "Уникальный текст"},
            {"text": "Одинаковый текст"},
        ])
        transform = ExactDeduplicationTransform(target_columns=["text"], num_proc=1)
        res = transform(ds)
        assert len(res) == 2
        assert res["text"] == ["Одинаковый текст", "Уникальный текст"]


class TestRAGTokenizationTransform:
    def test_indexing_mode_preserves_text(self):
        ds = Dataset.from_dict({"text": ["Test"]})
        transform = RAGTokenizationTransform(tokenizer=DummyRAGTokenizer(), mode="indexing", num_proc=1)
        res = transform(ds)
        assert "input_ids" in res.column_names
        assert "text" in res.column_names 
    """
    def test_contrastive_mode_handles_negatives(self):
        ds = Dataset.from_dict({"query": ["Q", "Q"], "positive_doc": ["P", "P"], "negative_doc": ["N", None]})
        transform = RAGTokenizationTransform(tokenizer=DummyRAGTokenizer(), mode="contrastive", num_proc=1, empty_doc_placeholder="")
        res = transform(ds)
        assert res["neg_input_ids"][1] is None"""


class TestLengthFilterTransform:
    def test_filters_long_sequences(self):
        ds = _make_dataset([
            {"input_ids": [1, 2, 3]},
            {"input_ids": [1, 2, 3, 4, 5, 6]},
        ])
        transform = LengthFilterTransform(max_length=4, column="input_ids", num_proc=1)
        res = transform(ds)
        assert len(res) == 1
        assert res[0]["input_ids"] == [1, 2, 3]