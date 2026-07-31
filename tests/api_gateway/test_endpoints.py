# tests/api_gateway/test_endpoints.py
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException


class TestChatStreamEndpoint:
    @pytest.mark.asyncio
    async def test_stream_returns_chunks(self, async_client, mock_orchestrator):
        # ИСПРАВЛЕНО: Явно инициализируем вложенные моки, которые мы добавили в архитектуру
        mock_orchestrator.build_prompt = AsyncMock(return_value="Готовый промпт")
        mock_orchestrator.llm_client = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield "Это моковый ответ."

        mock_orchestrator.llm_client.generate_stream = mock_stream

        async with async_client.stream(
            "POST", "/api/v1/chat/stream", json={"query": "Тестовый вопрос", "top_k": 3}
        ) as response:
            assert response.status_code == 200
            chunks = [chunk async for chunk in response.aiter_text()]
            assert "".join(chunks) == "Это моковый ответ."

        mock_orchestrator.build_prompt.assert_called_once_with(
            query="Тестовый вопрос", top_k=3, filters=None
        )

    @pytest.mark.asyncio
    async def test_stream_propagates_http_exceptions(self, async_client, mock_orchestrator):
        # ИСПРАВЛЕНО: Ошибка отлавливается на этапе формирования промпта
        mock_orchestrator.build_prompt = AsyncMock(
            side_effect=HTTPException(status_code=502, detail="RAG Error")
        )

        response = await async_client.post("/api/v1/chat/stream", json={"query": "Тестовый вопрос"})
        assert response.status_code == 502
        assert response.json()["detail"] == "RAG Error"

    @pytest.mark.asyncio
    async def test_stream_handles_unexpected_errors_in_generator(
        self, async_client, mock_orchestrator
    ):
        # ИСПРАВЛЕНО: Явно инициализируем моки
        mock_orchestrator.build_prompt = AsyncMock(return_value="Промпт")
        mock_orchestrator.llm_client = MagicMock()

        async def mock_stream_fatal(*args, **kwargs):
            yield "Начало ответа..."
            raise ValueError("Что-то пошло не так")

        mock_orchestrator.llm_client.generate_stream = mock_stream_fatal

        async with async_client.stream(
            "POST", "/api/v1/chat/stream", json={"query": "Тестовый вопрос"}
        ) as response:
            assert response.status_code == 200
            chunks = [chunk async for chunk in response.aiter_text()]
            full_text = "".join(chunks)

            assert "Начало ответа..." in full_text
            assert "[Ошибка при получении ответа]" in full_text
