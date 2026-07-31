# tests/application/test_orchestrator.py
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from src.application.orchestrator import RAGOrchestrator


class TestRAGOrchestrator:
    @pytest.fixture
    def mock_prompt_manager(self):
        manager = MagicMock()
        manager.render.return_value = "Готовый промпт"
        return manager

    @pytest.fixture
    def mock_llm_client(self):
        client = AsyncMock()

        async def mock_generate(*args, **kwargs):
            yield "Ответ"

        client.generate_stream = mock_generate
        return client

    @pytest.fixture
    def orchestrator(self, mock_llm_client, mock_prompt_manager):
        return RAGOrchestrator(
            rag_api_url="http://rag-api:8001",
            llm_client=mock_llm_client,
            prompt_manager=mock_prompt_manager,
            default_template="rag_qa",
            default_top_k=5,
        )

    @pytest.mark.asyncio
    async def test_build_prompt_success(self, orchestrator):
        """Проверяет успешный запрос в RAG API и формирование промпта."""
        mock_response = MagicMock()
        # ИСПРАВЛЕНО: Теперь структура мока полностью соответствует реальному RAG API
        mock_response.json.return_value = {"results": [{"metadata": {"text": "Текст документа 1"}}]}

        with patch.object(orchestrator.http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # ИСПРАВЛЕНО: Вызываем новый публичный метод build_prompt
            prompt = await orchestrator.build_prompt(
                query="Запрос", top_k=5, filters=None, template="rag_qa"
            )

            mock_post.assert_called_once_with(
                "http://rag-api:8001/api/v1/search",
                json={"query": "Запрос", "top_k": 5, "filters": None},
            )

            orchestrator.prompt_manager.render.assert_called_once()
            call_kwargs = orchestrator.prompt_manager.render.call_args[1]
            assert "Текст документа 1" in call_kwargs["context"]
            assert prompt == "Готовый промпт"

    @pytest.mark.asyncio
    async def test_build_prompt_raises_502_on_network_error(self, orchestrator):
        """Если RAG API недоступен, оркестратор должен бросать HTTP 502."""
        with patch.object(
            orchestrator.http_client, "post", side_effect=httpx.RequestError("Timeout")
        ):
            with pytest.raises(HTTPException) as exc:
                # ИСПРАВЛЕНО: Вызываем новый метод
                await orchestrator.build_prompt("Запрос", 5, None, "rag_qa")

            assert exc.value.status_code == 502
            assert "RAG Service is unavailable" in exc.value.detail
