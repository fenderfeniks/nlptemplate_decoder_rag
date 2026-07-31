# src/tg_bot/bot_webhook.py
"""Telegram-бот в режиме Webhook — интегрируется в FastAPI.

Диспетчер инициализируется один раз при старте приложения.
Оркестратор пробрасывается из ``app.state.ml_models`` через lifespan,
чтобы бот и RAG API разделяли одни загруженные модели.
"""

import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.redis import RedisStorage

from src.tg_bot.handlers.chat import router as chat_router


logger = logging.getLogger(__name__)

# Redis для FSM: обязателен при нескольких воркерах uvicorn.
# Без Redis каждый воркер имеет своё state-хранилище — сессии теряются
# при переключении между воркерами.
redis_url = os.getenv("REDIS_URL")
if redis_url:
    logger.info("FSM storage: Redis (%s).", redis_url.split("@")[-1])  # скрываем пароль
    storage = RedisStorage.from_url(redis_url)
else:
    logger.warning(
        "REDIS_URL не задан → MemoryStorage. При нескольких воркерах uvicorn сессии будут теряться!"
    )
    storage = MemoryStorage()

dp = Dispatcher(storage=storage)
dp.include_router(chat_router)


def setup_dispatcher(orchestrator=None, prompt_manager=None, cfg=None) -> None:
    """Инжектирует зависимости в диспетчер из FastAPI lifespan.

    Вызывается из ``server.py`` после инициализации RAG-стека:

    .. code-block:: python

        from src.tg_bot.bot_webhook import dp, setup_dispatcher
        setup_dispatcher(
            orchestrator=app.state.ml_models["orchestrator"],
            prompt_manager=prompt_manager,
            cfg=cfg,
        )

    Args:
        orchestrator: Инстанс RAGOrchestrator с загруженными моделями.
        prompt_manager: Инстанс PromptManager для рендеринга шаблонов.
        cfg: Hydra DictConfig для доступа к параметрам из хендлеров.
    """
    if orchestrator is not None:
        dp["orchestrator"] = orchestrator
        logger.info("Webhook-бот: оркестратор подключён.")
    if prompt_manager is not None:
        dp["prompt_manager"] = prompt_manager
    if cfg is not None:
        dp["cfg"] = cfg
    dp["api_url"] = None  # webhook всегда использует оркестратор, не HTTP


def get_webhook_bot(token: str) -> Bot:
    """Создаёт экземпляр Bot для обработки вебхуков.

    Args:
        token: Telegram Bot API токен.

    Returns:
        Инициализированный ``Bot``.
    """
    logger.info("Инициализация Bot для режима Webhook.")
    return Bot(token=token)
