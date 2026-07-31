from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.decoder_pipeline.api.rest.dependencies import get_generator, get_prompt_manager
from src.decoder_pipeline.api.rest.endpoints import generate, health


@pytest.fixture
def mock_generator():
    client = AsyncMock()
    # ИСПРАВЛЕНО: Устанавливаем возвращаемое значение самому объекту, а не его __call__
    client.return_value = [{"prompt": "test", "generated_text": "Сгенерированный ответ"}]

    # Настраиваем стриминг
    client.generate_stream = MagicMock()

    async def mock_stream(*args, **kwargs):
        yield "Стрим "
        yield "работает!"

    client.generate_stream.return_value = mock_stream()
    return client


@pytest.fixture
def mock_prompt_manager():
    manager = MagicMock()
    manager.render.return_value = "Готовый промпт для LLM"
    return manager


@pytest.fixture
def test_app(mock_generator, mock_prompt_manager):
    app = FastAPI()
    app.state.ml_models = {"generator": mock_generator}
    app.state.prompt_manager = mock_prompt_manager
    app.state.config = MagicMock()
    app.state.config.api.get.return_value = {"default_template": "test", "static_context": ""}

    app.include_router(health.router)
    app.include_router(generate.router)

    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_prompt_manager] = lambda: mock_prompt_manager
    return app


@pytest.fixture
async def async_client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check_ok(self, async_client):
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, async_client, test_app):
        test_app.state.ml_models = {}
        test_app.state.prompt_manager = None

        response = await async_client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"


class TestGenerateEndpoint:
    @pytest.mark.asyncio
    async def test_generate_text_success(self, async_client, mock_prompt_manager, mock_generator):
        response = await async_client.post("/api/v1/generate", json={"prompt": "Привет"})

        assert response.status_code == 200
        assert response.json()["generated_text"] == "Сгенерированный ответ"
        # ИСПРАВЛЕНО: Проверяем вызов самого мока
        mock_generator.assert_called_once_with("Готовый промпт для LLM")

    @pytest.mark.asyncio
    async def test_generate_text_llm_failure(self, async_client, mock_generator):
        # ИСПРАВЛЕНО: Назначаем side_effect самому моку
        mock_generator.side_effect = Exception("Connection Refused")
        response = await async_client.post("/api/v1/generate", json={"prompt": "Привет"})

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, async_client, mock_generator):
        async with async_client.stream(
            "POST", "/api/v1/generate/stream", json={"prompt": "Привет"}
        ) as response:
            assert response.status_code == 200
            chunks = [chunk async for chunk in response.aiter_text()]
            assert "".join(chunks) == "Стрим работает!"

        mock_generator.generate_stream.assert_called_once_with("Готовый промпт для LLM")
