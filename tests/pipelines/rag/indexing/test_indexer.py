# tests/pipelines/rag/indexing/test_indexer.py
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.pipelines.rag.indexing.indexer import KnowledgeBaseIndexer


@pytest.fixture
def dummy_indexer():
    embedder = MagicMock()
    embedder.encode.return_value = np.ones((2, 4), dtype=np.float32)

    store = MagicMock()
    store.existing_doc_ids = set()
    store.ntotal = 0  # Исправлено: прямое обращение к свойству

    # Исправлено: передаем store и lsh=None
    return KnowledgeBaseIndexer(embedder, store=store, lsh=None, push_batch_size=2)


class TestKnowledgeBaseIndexer:
    def test_generate_doc_id(self):
        """Генерация детерминированного SHA-256 (16 символов)."""
        doc_id = KnowledgeBaseIndexer._generate_doc_id(
            "Текст", {"url": "http://test", "title": "Заголовок"}
        )
        assert len(doc_id) == 16
        assert isinstance(doc_id, str)

        doc_id2 = KnowledgeBaseIndexer._generate_doc_id(
            "Текст", {"url": "http://test", "title": "Заголовок"}
        )
        assert doc_id == doc_id2

    def test_exact_duplicate_detection(self, dummy_indexer):
        """Точные дубликаты определяются по ID в существующих или новых сетах."""
        assert dummy_indexer._is_exact_duplicate("id1", {"id1"}, set()) is True
        assert dummy_indexer._is_exact_duplicate("id2", set(), {"id2"}) is True
        assert dummy_indexer._is_exact_duplicate("id3", {"id1"}, {"id2"}) is False

    def test_index_dataloader_skips_duplicates(self, dummy_indexer):
        """Проверка пропуска точных дубликатов и корректного буферизованного пуша в БД."""
        dataloader = [
            {
                "input_ids": [1, 2],
                "text": ["Документ 1", "Документ 1"],
                "metadata": [{"id": 1}, {"id": 1}],
            }
        ]

        dummy_indexer.index_dataloader(dataloader)

        call_args = dummy_indexer.embedder.encode.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] == "Документ 1"

        # Исправлено: проверяем вызов insert через store
        dummy_indexer.store.insert.assert_called_once()
        insert_args = dummy_indexer.store.insert.call_args[0]
        assert len(insert_args[1]) == 1
        assert insert_args[1][0]["text"] == "Документ 1"
        assert "doc_id" in insert_args[1][0]
