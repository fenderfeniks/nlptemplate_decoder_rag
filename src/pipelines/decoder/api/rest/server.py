import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.pipelines.decoder.api.rest.endpoints import generate, health
from src.pipelines.decoder.api.rest.limiter import limiter
from src.pipelines.decoder.api.rest.middlewares import setup_middlewares
from src.pipelines.decoder.core.prompts.manager import PromptManager
from src.pipelines.decoder.inference.inference import LLMGenerationClient
from src.tg_bot.bot_webhook import get_webhook_bot


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Читаем ключ один раз при старте приложения, а не на каждый запрос.
# Если переменная не задана — API работает без аутентификации (с предупреждением).
_EXPECTED_API_KEY: str | None = os.getenv("API_KEY")
if not _EXPECTED_API_KEY:
    logger.warning(
        "Переменная окружения API_KEY не задана. "
        "Все запросы к защищённым эндпоинтам будут пропускаться без проверки ключа."
    )


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if _EXPECTED_API_KEY and api_key != _EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key or ""


def create_app() -> FastAPI:
    load_dotenv()
    config_dir = Path(__file__).resolve().parents[4] / "configs"

    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass  # Уже очищен или не инициализирован

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="main")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ml_models = {}
        app.state.prompt_manager = PromptManager(templates=cfg.get("prompts", {}))

        logger.info("Подключение к внешнему LLM-серверу...")
        llm_url = cfg.services.get("llm_api_url", "http://localhost:8000/v1")
        generator = LLMGenerationClient(
            api_base=llm_url,
            temperature=cfg.api.get("generation", {}).get("temperature", 0.7),
        )
        app.state.ml_models["generator"] = generator

        # Telegram-бот: токен не попадает в логи
        bot_token: str | None = os.getenv("TG_BOT_TOKEN") or cfg.api.telegram.get("bot_token")
        if bot_token:
            bot = get_webhook_bot(bot_token)
            app.state.tg_bot = bot
            try:
                webhook_url = cfg.api.telegram.webhook_url
                await bot.set_webhook(url=webhook_url, drop_pending_updates=True)
                logger.info("Telegram webhook установлен.")
            except Exception as e:
                logger.warning("Не удалось установить вебхук Telegram: %s", e)
        else:
            logger.info("TG_BOT_TOKEN не задан — Telegram-бот не запускается.")

        yield

        if hasattr(app.state, "tg_bot"):
            await app.state.tg_bot.delete_webhook()
            await app.state.tg_bot.session.close()

        app.state.ml_models.clear()

    app = FastAPI(
        title=cfg.api.title,
        description=cfg.api.description,
        version=cfg.api.version,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.state.config = cfg

    setup_middlewares(app, cors_origins=list(cfg.api.cors_origins))

    app.include_router(health.router)
    app.include_router(generate.router, dependencies=[Depends(verify_api_key)])

    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

    return app


app = create_app()
