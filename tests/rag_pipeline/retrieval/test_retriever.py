# tests/rag_pipeline/retrieval/test_retriever.py
from unittest.mock import MagicMock

import pytest

from src.rag_pipeline.retrieval.retriever import BaseRetriever


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Возвращаем "векторы" в виде списка/массива
    embedder.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    return embedder


@pytest.fixture
def mock_vector_db():
    db = MagicMock()
    db.index.ntotal = 100
    # Имитация сырых результатов от FAISS
    db.search.return_value = [
        [
            {"score": 0.95, "metadata": {"doc_id": "1"}},
            {"score": 0.85, "metadata": {"doc_id": "2"}},
            {"score": 0.70, "metadata": {"doc_id": "3"}},
        ],
        [{"score": 0.99, "metadata": {"doc_id": "4"}}],
    ]
    return db


class TestBaseRetriever:
    def test_search_returns_single_list(self, mock_embedder, mock_vector_db):
        """Метод search должен возвращать плоский список для одного запроса."""
        retriever = BaseRetriever(embedder=mock_embedder, vector_db=mock_vector_db)
        result = retriever.search(query="Что такое RAG?", top_k=3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["score"] == 0.95

    def test_batch_search_raises_on_empty_queries(self, mock_embedder, mock_vector_db):
        """При пустом списке запросов batch_search должен выбрасывать ValueError."""
        retriever = BaseRetriever(embedder=mock_embedder, vector_db=mock_vector_db)

        with pytest.raises(ValueError, match="не может быть пустым списком"):
            retriever.batch_search(queries=[])

    def test_batch_search_returns_empty_when_db_empty(self, mock_embedder, mock_vector_db):
        """Если база пуста, ретривер не вызывает энкодер и возвращает пустые списки."""
        mock_vector_db.index.ntotal = 0
        retriever = BaseRetriever(embedder=mock_embedder, vector_db=mock_vector_db)

        result = retriever.batch_search(queries=["Q1", "Q2"])

        assert result == [[], []]
        mock_embedder.encode.assert_not_called()

    def test_score_threshold_filters_results(self, mock_embedder, mock_vector_db):
        """Результаты со score ниже порога должны отбрасываться."""
        retriever = BaseRetriever(embedder=mock_embedder, vector_db=mock_vector_db)
        result = retriever.search(query="RAG", top_k=3, score_threshold=0.80)

        assert len(result) == 2
        assert all(r["score"] >= 0.80 for r in result)
