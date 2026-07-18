"""
Dependency Injection (DI) контейнер для FastAPI.
Обеспечивает безопасный доступ к ML-моделям и сервисам для всех эндпоинтов.
"""

from fastapi import HTTPException, Request

from src.core.models.generation import HFTextGenerator
from src.core.models.promts import PromptManager
from src.core.rag.retriever import RAGRetriever


def get_generator(request: Request) -> HFTextGenerator:
    """Достает генератор текстов из глобальной памяти сервера."""
    generator = request.app.state.ml_models.get("generator")
    if not generator:
        raise HTTPException(status_code=503, detail="Модель генерации еще не загружена в память.")
    return generator


def get_retriever(request: Request) -> RAGRetriever:
    """Достает RAG-ретривер из глобальной памяти сервера."""
    retriever = request.app.state.ml_models.get("retriever")
    if not retriever:
        raise HTTPException(status_code=503, detail="Векторная база еще не инициализирована.")
    return retriever


def get_prompt_manager() -> PromptManager:
    """
    Провайдер для PromptManager.
    Так как класс содержит только статические методы, мы просто возвращаем сам класс.
    В будущем сюда можно добавить логику (например, A/B тестирование разных промптов).
    """
    return PromptManager
