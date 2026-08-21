# tests/pipelines/rag/retrieval/test_retriever.py
from unittest.mock import MagicMock

import pytest

from src.pipelines.rag.inference.retriever import BaseRetriever


@pytest.fixture
def dummy_components():
    embedder = MagicMock()
    vector_db = MagicMock()
    vector_db.ntotal = 100
    return embedder, vector_db


class TestBaseRetriever:
    def test_batch_search_empty_queries(self, dummy_components):
        """Проверка защиты от пустого списка запросов."""
        embedder, vector_db = dummy_components
        retriever = BaseRetriever(embedder, vector_db)

        with pytest.raises(ValueError, match="не может быть пустым списком"):
            retriever.batch_search([])

    def test_batch_search_empty_db(self, dummy_components):
        """Если база пуста (ntotal == 0), возвращаются пустые списки."""
        embedder, vector_db = dummy_components
        vector_db.ntotal = 0  # Исправлено
        retriever = BaseRetriever(embedder, vector_db)

        result = retriever.batch_search(["query1", "query2"])
        assert result == [[], []]
        embedder.encode.assert_not_called()

    def test_batch_search_embedder_runtime_error(self, dummy_components):
        """При RuntimeError в эмбеддере поиск не падает, а возвращает пустые списки."""
        embedder, vector_db = dummy_components
        embedder.encode.side_effect = RuntimeError("CUDA OOM")
        retriever = BaseRetriever(embedder, vector_db)

        result = retriever.batch_search(["query1"])
        assert result == [[]]

    def test_batch_search_with_threshold(self, dummy_components):
        """Проверка фильтрации кандидатов по score_threshold."""
        embedder, vector_db = dummy_components
        vector_db.search.return_value = [
            [{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.5}],
            [{"doc_id": 3, "score": 0.4}],
        ]
        retriever = BaseRetriever(embedder, vector_db)

        result = retriever.batch_search(["q1", "q2"], score_threshold=0.8)

        assert len(result) == 2
        assert len(result[0]) == 1
        assert result[0][0]["doc_id"] == 1
        assert len(result[1]) == 0

    def test_search_single_query(self, dummy_components):
        """Метод search должен пробрасывать запрос в batch_search и возвращать первый элемент."""
        embedder, vector_db = dummy_components
        vector_db.search.return_value = [[{"doc_id": 1, "score": 0.9}]]
        retriever = BaseRetriever(embedder, vector_db)

        result = retriever.search("single query")
        assert result == [{"doc_id": 1, "score": 0.9}]
