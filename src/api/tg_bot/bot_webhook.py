"""
Инициализация диспетчера Telegram-бота для работы через Webhooks.
Интегрируется напрямую в веб-сервер FastAPI.
"""

import logging

from aiogram import Bot, Dispatcher

from src.api.tg_bot.handlers.chat import router as chat_router


logger = logging.getLogger(__name__)

# Создаем глобальный диспетчер и регистрируем хэндлеры
dp = Dispatcher()
dp.include_router(chat_router)


def get_webhook_bot(token: str) -> Bot:
    """Инициализирует экземпляр бота для вебхуков."""
    logger.info("Инициализация экземпляра Bot для Webhooks...")
    return Bot(token=token)
