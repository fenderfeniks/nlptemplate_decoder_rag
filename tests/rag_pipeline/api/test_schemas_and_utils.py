# tests/rag_pipeline/api/test_schemas_and_utils.py
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.rag_pipeline.api.rest.limiter import get_real_ip
from src.rag_pipeline.api.schemas import Document, SearchRequest


class TestSearchRequestSchema:
    def test_valid_request(self):
        req = SearchRequest(query="Нормальный запрос", top_k=10)
        assert req.query == "Нормальный запрос"
        assert req.top_k == 10

    def test_query_too_short_raises(self):
        """Запросы короче 2 символов отбрасываются."""
        with pytest.raises(ValidationError):
            SearchRequest(query="a")

    def test_top_k_bounds(self):
        """Проверка ограничений для поля top_k (ge=1, le=50)."""
        with pytest.raises(ValidationError):
            SearchRequest(query="Запрос", top_k=0)

        with pytest.raises(ValidationError):
            SearchRequest(query="Запрос", top_k=51)


class TestDocumentSchema:
    def test_score_bounds(self):
        """Поле score должно быть в пределах от 0.0 до 1.0."""
        Document(score=1.0, metadata={})

        with pytest.raises(ValidationError):
            Document(score=1.5, metadata={})

        with pytest.raises(ValidationError):
            Document(score=-0.1, metadata={})


class TestLimiterUtils:
    def test_get_real_ip_with_x_forwarded_for(self):
        """Извлечение IP должно приоритизировать X-Forwarded-For."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "192.168.1.1, 10.0.0.1"

        ip = get_real_ip(mock_request)
        assert ip == "192.168.1.1"
