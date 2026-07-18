"""
Локальный Telegram-бот (Long Polling).
Запускается как независимый процесс и общается с FastAPI по HTTP.
"""

import asyncio
import logging
import os

import hydra
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Импортируем наш универсальный роутер хэндлеров
from src.api.tg_bot.handlers.chat import router as chat_router


# Загружаем переменные окружения из .env
load_dotenv()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Разрешаем ссылки в конфигах
    OmegaConf.resolve(cfg)

    # 1. Получаем токен (приоритет у .env, затем у Гидры)
    bot_token = os.getenv("TG_BOT_TOKEN") or cfg.api.telegram.bot_token
    if not bot_token:
        raise ValueError("Критическая ошибка: TG_BOT_TOKEN не найден ни в .env, ни в конфигурации!")

    bot = Bot(token=bot_token)
    dp = Dispatcher()

    # Подключаем роутер с хэндлерами
    dp.include_router(chat_router)

    # Собираем URL локального FastAPI для отправки запросов
    api_url = f"{cfg.api.domain}/chat/generate"

    async def start_polling():
        logger.info("Удаление старых вебхуков...")
        await bot.delete_webhook(drop_pending_updates=True)

        logger.info("Запуск локального бота в режиме Polling...")
        # aiogram 3 позволяет прокидывать любые переменные в старт-поллинг.
        # Они автоматически прилетят в аргументы функций-хэндлеров по именам!
        await dp.start_polling(bot, cfg=cfg, api_url=api_url)

    # Запускаем асинцио-цикл внутри синхронной функции Hydra
    asyncio.run(start_polling())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
