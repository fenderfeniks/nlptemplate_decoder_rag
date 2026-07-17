"""
Главный файл веб-сервера FastAPI.
Управляет жизненным циклом моделей, настраивает роутеры, CORS,
а также регистрирует и обслуживает Telegram-вебхуки на проде.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from hydra import initialize, compose
from hydra.utils import instantiate
from aiogram import types

from prometheus_fastapi_instrumentator import Instrumentator

from src.api.rest.endpoints import health, chat
from src.api.rest.middlewares import setup_middlewares

# Импортируем инфраструктуру бота для вебхуков
from src.api.tg_bot.bot_webhook import dp, get_webhook_bot

logger = logging.getLogger(__name__)

# Загружаем конфигурацию Hydra один раз при импорте модуля
with initialize(version_base="1.3", config_path="../../../configs"):
    cfg = compose(config_name="main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом тяжелых объектов (VRAM и Webhooks)."""
    app.state.ml_models = {}
    
    # 1. Загрузка тяжелых ML компонентов
    logger.info("Загрузка ML моделей в видеопамять...")
    tokenizer = instantiate(cfg.model.tokenizer).build()
    model = instantiate(cfg.model.builder, tokenizer=tokenizer).build()
    
    generator = instantiate(cfg.model.generation, model=model, tokenizer=tokenizer)
    retriever = instantiate(cfg.rag.retriever)
    prompt_manager = instantiate(cfg.model_module.get("prompt_manager_cfg", None)) 
    # Если PromptManager не в конфиге, возьмем класс напрямую
    if not prompt_manager:
        from src.core.models.promts import PromptManager
        prompt_manager = PromptManager

    app.state.ml_models["generator"] = generator
    app.state.ml_models["retriever"] = retriever
    app.state.ml_models["prompt_manager"] = prompt_manager
    
    # 2. Инициализация Telegram Webhook
    bot_token = os.getenv("TG_BOT_TOKEN") or cfg.api.telegram.bot_token
    if bot_token:
        bot = get_webhook_bot(bot_token)
        app.state.tg_bot = bot
        
        # Настраиваем путь вебхука из конфига Гидры
        webhook_url = cfg.api.telegram_webhook.url
        logger.info(f"Установка Telegram Webhook на адрес: {webhook_url}")
        await bot.set_webhook(url=webhook_url, drop_pending_updates=True)
    else:
        logger.warning("TG_BOT_TOKEN не найден. Запуск сервера без поддержки Telegram Webhook.")

    yield # Сервер запущен и принимает HTTP-запросы
    
    # 3. Очистка при выключении
    logger.info("Остановка сервера. Очистка ресурсов...")
    if "tg_bot" in app.state:
        logger.info("Удаление Telegram Webhook...")
        await app.state.tg_bot.delete_webhook()
        await app.state.tg_bot.session.close()
        
    app.state.ml_models.clear()


# Инициализация FastAPI приложения
app = FastAPI(
    title=cfg.api.title,
    description=cfg.api.description,
    version=cfg.api.version,
    lifespan=lifespan
)

# Настройка CORS и логирования времени генерации
setup_middlewares(app, cors_origins=list(cfg.api.cors_origins))

# Подключение REST API роутеров
app.include_router(health.router)
app.include_router(chat.router)

Instrumentator(
    should_group_status_codes=False, 
    should_ignore_untemplated=True
).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

# 4. Эндпоинт-приемник для Webhook
@app.post(cfg.api.telegram_webhook.path, include_in_schema=False)
async def telegram_webhook_endpoint(update: dict):
    """
    Принимает зашифрованные пакеты от серверов Telegram и отправляет их в диспетчер бота.
    """
    bot = app.state.tg_bot
    if not bot:
        raise HTTPException(status_code=503, detail="Telegram bot service is unavailable")
        
    telegram_update = types.Update(**update)
    
    # Внедряем зависимости напрямую в вебхук-диспетчер aiogram!
    # Они автоматически попадут в аргументы нашего хэндлера process_chat_message
    await dp.feed_update(
        bot,
        update=telegram_update,
        cfg=cfg,
        generator=app.state.ml_models["generator"],
        retriever=app.state.ml_models["retriever"],
        prompt_manager=app.state.ml_models["prompt_manager"]
    )
    return {"status": "ok"}