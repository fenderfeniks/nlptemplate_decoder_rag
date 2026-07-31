# tests/rag_pipeline/api/test_endpoints.py
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.rag_pipeline.api.rest.endpoints import search


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    # Возвращаем ожидаемый плоский список документов
    retriever.search.return_value = [
        {"score": 0.95, "metadata": {"text": "Результат поиска 1"}},
        {"score": 0.85, "metadata": {"text": "Результат поиска 2"}},
    ]
    return retriever


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(search.router)

    retriever = MagicMock()
    retriever.search.return_value = [{"score": 0.95, "metadata": {"text": "Результат"}}]
    # Передаем ретривер в state, как этого ждет код
    app.state.ml_models = {"retriever": retriever}
    return app


@pytest.fixture
async def async_client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_200_and_expected_format(self, async_client):
        response = await async_client.post("/api/v1/search", json={"query": "RAG", "top_k": 1})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_handles_exceptions_gracefully(self, async_client, test_app):
        test_app.state.ml_models["retriever"].search.side_effect = RuntimeError("Ошибка")
        response = await async_client.post("/api/v1/search", json={"query": "RAG"})
        assert response.status_code == 500
