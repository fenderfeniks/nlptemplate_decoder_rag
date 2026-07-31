# tests/rag_pipeline/indexing/test_indexer.py
from unittest.mock import MagicMock

import pytest
import torch

from src.rag_pipeline.indexing.indexer import KnowledgeBaseIndexer


class TestKnowledgeBaseIndexer:
    def test_invalid_precision_raises(self):
        """Инциализация с неверным precision должна падать."""
        with pytest.raises(ValueError, match="Недопустимое значение precision"):
            KnowledgeBaseIndexer(
                model=MagicMock(), pooler=MagicMock(), vector_db=MagicMock(), precision="int8"
            )

    def test_generate_doc_id_is_deterministic(self):
        """Проверка детерминированности MD5-хэша на основе текста и меты."""
        indexer = KnowledgeBaseIndexer(
            model=MagicMock(), pooler=MagicMock(), vector_db=MagicMock(), precision="fp32"
        )
        text = "Тестовый документ"
        meta = {"url": "https://test.com", "title": "Test Title"}

        id1 = indexer._generate_doc_id(text, meta)
        id2 = indexer._generate_doc_id(text, meta)

        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) == 32  # Длина MD5

    def test_to_tensor_indices(self):
        """Проверка корректной конвертации индексов в torch.long."""
        indexer = KnowledgeBaseIndexer(
            model=MagicMock(), pooler=MagicMock(), vector_db=MagicMock(), precision="fp32"
        )
        indices = [0, 2, 5]
        tensor = indexer._to_tensor_indices(indices, batch_size=10)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
