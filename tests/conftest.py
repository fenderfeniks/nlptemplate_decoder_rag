# tests/conftest.py
import asyncio
import gc
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
import torch
from httpx import ASGITransport, AsyncClient


os.environ["PROJECT_ROOT"] = os.getcwd()
os.environ.setdefault("ENVIRONMENT", "testing")

from src.api_gateway.dependencies import get_orchestrator
from src.api_gateway.server import create_gateway_app
from src.application.orchestrator import RAGOrchestrator


# --- Фикстура очистки мусора и асинхронных задач ---
@pytest_asyncio.fixture(autouse=True)
async def cleanup_memory_and_tasks():
    """Принудительно отменяет фоновые задачи и чистит VRAM после каждого теста."""
    yield

    # 1. Находим все задачи, кроме текущей (самого Pytest)
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    # 2. Отменяем их
    for task in pending:
        task.cancel()

    # 3. Даем Event Loop возможность обработать отмену
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    # 4. Жесткая сборка мусора
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Остальные фикстуры ---
@pytest.fixture(scope="function")
def test_app():
    return create_gateway_app()


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    orchestrator = AsyncMock(spec=RAGOrchestrator)
    return orchestrator


@pytest_asyncio.fixture
async def async_client(test_app, mock_orchestrator):
    test_app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
