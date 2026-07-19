# src/api/rest/server.py
import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
from aiogram import types
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from prometheus_fastapi_instrumentator import Instrumentator

# ИСПРАВЛЕНИЕ: Удален импорт Limiter и get_remote_address, чтобы не затирать переменную
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.rest.endpoints import chat, health
from src.api.rest.limiter import limiter
from src.api.rest.middlewares import setup_middlewares
from src.api.tg_bot.bot_webhook import dp, get_webhook_bot


logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key


def create_app(load_ml: bool = True) -> FastAPI:
    load_dotenv()
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="main")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ml_models = {}

        if load_ml:
            logger.info("Загрузка ML моделей в видеопамять...")
            tokenizer = instantiate(cfg.model.tokenizer).build()
            model = instantiate(cfg.model.builder, tokenizer=tokenizer).build()
            generator = instantiate(cfg.model.generation, model=model, tokenizer=tokenizer)

            retriever_cfg = cfg.get("rag", {}).get("retriever")
            if retriever_cfg:
                retriever = instantiate(retriever_cfg)
            else:
                from src.core.rag.retriever import RAGRetriever

                # ИСПРАВЛЕНИЕ: Передаем обязательный persist_dir из конфига
                retriever = RAGRetriever(persist_dir=cfg.rag.persist_dir)

            prompt_manager = instantiate(cfg.model_module.get("prompt_manager_cfg", None))
            if not prompt_manager:
                from src.core.models.promts import PromptManager

                prompt_manager = PromptManager

            app.state.ml_models["generator"] = generator
            app.state.ml_models["retriever"] = retriever
            app.state.ml_models["prompt_manager"] = prompt_manager

            bot_token = os.getenv("TG_BOT_TOKEN") or cfg.api.telegram.bot_token
            if bot_token:
                bot = get_webhook_bot(bot_token)
                app.state.tg_bot = bot
                webhook_url = cfg.api.telegram.webhook_url
                await bot.set_webhook(url=webhook_url, drop_pending_updates=True)

        yield

        if load_ml:
            if "tg_bot" in app.state:
                await app.state.tg_bot.delete_webhook()
                await app.state.tg_bot.session.close()

            app.state.ml_models.clear()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # ИСПРАВЛЕНИЕ: Удалена глобальная зависимость verify_api_key
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

    # ИСПРАВЛЕНИЕ: Healthcheck остается открытым для K8s
    app.include_router(health.router)
    # ИСПРАВЛЕНИЕ: Защищаем API-ключом только боевые эндпоинты
    app.include_router(chat.router, dependencies=[Depends(verify_api_key)])

    Instrumentator(should_group_status_codes=False, should_ignore_untemplated=True).instrument(
        app
    ).expose(app, include_in_schema=False, endpoint="/metrics")

    @app.post(cfg.api.telegram_webhook.path, include_in_schema=False)
    async def telegram_webhook_endpoint(update: dict):
        bot = app.state.tg_bot
        if not bot:
            raise HTTPException(status_code=503, detail="Telegram bot service is unavailable")
        await dp.feed_update(
            bot,
            update=types.Update(**update),
            cfg=cfg,
            generator=app.state.ml_models.get("generator"),
            retriever=app.state.ml_models.get("retriever"),
            prompt_manager=app.state.ml_models.get("prompt_manager"),
        )
        return {"status": "ok"}

    return app


app = create_app(load_ml=True)
