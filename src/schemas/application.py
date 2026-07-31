# src/schemas/application.py
"""Схемы конфигурации для application-слоя:
Telegram-бот, RAGOrchestrator и LlamaIndex-режим.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TgBotMessagesConfig:
    """Текстовые сообщения бота — меняются без правки кода."""

    start: str = "Привет! Я RAG-ассистент. Задайте вопрос — найду ответ в базе знаний."
    processing: str = "✨ Ищу информацию и генерирую ответ..."
    error: str = "Произошла ошибка при обработке запроса. Попробуйте ещё раз."
    empty_result: str = "К сожалению, не нашёл релевантных документов по вашему запросу."


@dataclass
class TgBotConfig:
    """Конфигурация Telegram-бота."""

    # Режим работы:
    # false → HTTP-фолбек к API (dev, модели не грузятся в процесс бота)
    # true  → RAGOrchestrator в памяти (prod, без отдельного API-сервера)
    use_orchestrator: bool = False

    # Шаблон промпта из PromptManager
    rag_template: str = "rag_qa"

    # Параметры ретривала
    top_k: int = 5
    score_threshold: Optional[float] = None  # null = без порога

    # Интервал редактирования сообщения при стриминге (Telegram rate limit ~1/сек)
    stream_edit_interval: float = 1.5

    messages: TgBotMessagesConfig = field(default_factory=TgBotMessagesConfig)


@dataclass
class OrchestratorConfig:
    """Статические параметры RAGOrchestrator.

    retriever и generator пробрасываются из кода как runtime-аргументы
    (требуют загруженных весов — не могут быть instantiate напрямую).
    """

    default_template: str = "${tg_bot.rag_template}"
    default_top_k: int = "${tg_bot.top_k}"  # type: ignore[assignment]


@dataclass
class LlamaIndexConfig:
    """Параметры для run_llamaindex.py."""

    # Число документов для query_engine
    similarity_top_k: int = 3

    # Тестовый запрос; переопределяется через CLI:
    # application.llamaindex.test_query="твой вопрос"
    test_query: str = "Какие существуют архитектурные паттерны для масштабирования RAG?"


@dataclass
class ApplicationConfig:
    """Корневая схема application-слоя."""

    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    llamaindex: LlamaIndexConfig = field(default_factory=LlamaIndexConfig)
