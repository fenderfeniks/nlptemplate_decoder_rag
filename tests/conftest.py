# tests/conftest.py
import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# Настройка окружения
os.environ["PROJECT_ROOT"] = os.getcwd()
os.environ.setdefault("PROJECT_NAME", "Test NLP API")
os.environ.setdefault("PROJECT_VERSION", "0.1.0")
os.environ.setdefault("PROJECT_DESCRIPTION", "Test API")
os.environ.setdefault("API_PORT", "8000")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test_token")
# ИСПРАВЛЕНИЕ: Добавлен мок для токена телеграма, чтобы OmegaConf.resolve не падал
os.environ.setdefault("TG_BOT_TOKEN", "test_bot_token")

# Импортируем фабрику, а не глобальный app
from src.api.rest.dependencies import get_generator, get_prompt_manager, get_retriever
from src.api.rest.server import create_app
from src.core.models.promts import PromptManager


@pytest.fixture(scope="function")
def test_app():
    """Фабрика: создает чистое приложение для каждого теста без загрузки моделей."""
    app = create_app(load_ml=False)
    # Инициализируем state, чтобы не было ошибок доступа
    app.state.ml_models = {}
    return app


@pytest.fixture
def mock_generator() -> MagicMock:
    generator = MagicMock(name="HFTextGenerator")
    generator.generation_kwargs = {"max_new_tokens": 256}
    generator.generate.return_value = ["Мок-ответ модели."]
    return generator


@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock(name="RAGRetriever")
    retriever.retrieve_context.return_value = "Контекст из FAISS."
    return retriever


@pytest.fixture
def override_ml_deps(test_app, mock_generator, mock_retriever):
    """Переопределяем зависимости для экземпляра приложения test_app."""
    test_app.dependency_overrides[get_generator] = lambda: mock_generator
    test_app.dependency_overrides[get_retriever] = lambda: mock_retriever
    test_app.dependency_overrides[get_prompt_manager] = lambda: PromptManager

    # Также обновляем state, если код эндпоинтов берет моки оттуда напрямую
    test_app.state.ml_models = {
        "generator": mock_generator,
        "retriever": mock_retriever,
        "prompt_manager": PromptManager,
    }

    yield test_app

    test_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(test_app, override_ml_deps):
    """Клиент использует свежий test_app с переопределенными зависимостями."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
