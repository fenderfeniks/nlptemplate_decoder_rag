# tests/api_gateway/test_schemas.py
import pytest
from pydantic import ValidationError

from src.api_gateway.schemas import ChatRequest


class TestChatRequestSchema:
    def test_valid_request(self):
        req = ChatRequest(query="Что такое RAG?", top_k=3)
        assert req.query == "Что такое RAG?"
        assert req.top_k == 3

    def test_query_too_short_raises(self):
        """Запрос не может быть пустым (min_length=1)."""
        with pytest.raises(ValidationError):
            ChatRequest(query="")

    def test_query_too_long_raises(self):
        """Запрос ограничивается 1500 символами."""
        with pytest.raises(ValidationError):
            ChatRequest(query="A" * 1501)

    def test_top_k_bounds(self):
        """Проверка ограничений top_k от 1 до 10."""
        with pytest.raises(ValidationError):
            ChatRequest(query="Тест", top_k=0)

        with pytest.raises(ValidationError):
            ChatRequest(query="Тест", top_k=11)
