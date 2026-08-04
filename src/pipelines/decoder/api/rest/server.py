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
from src.tools.storage.resolver import ArtifactResolver


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Проверяет API-ключ из заголовка X-API-Key."""
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key or ""


def create_app() -> FastAPI:
    """Фабрика FastAPI-приложения для decoder (LLM generation) пайплайна."""
    load_dotenv()

    if not os.getenv("API_KEY"):
        logger.warning(
            "API_KEY не задан — все запросы к защищённым эндпоинтам "
            "пропускаются без проверки ключа."
        )

    config_dir = Path(__file__).resolve().parents[4] / "configs"

    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="main")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ml_models = {}
        app.state.prompt_manager = PromptManager(templates=cfg.get("prompts", {}))

        # 1. Резолвинг артефактов (Скачивание весов для vLLM)
        logger.info("Синхронизация артефактов модели (Decoder)...")
        router = hydra.utils.instantiate(cfg.storage_router)
        cache_base = Path(cfg.paths.model_dir) / "decoder_cache"
        resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

        manifest_uri = os.getenv(
            "MANIFEST_URI", "local://./prod_storage/manifests/decoder_manifest.json"
        )

        try:
            # FastAPI скачивает веса в cache_base. Если vLLM и FastAPI работают
            # в одном окружении (или через shared volume в Docker), vLLM подхватит эти файлы.
            resolver.resolve_and_patch(cfg, manifest_uri, pipeline_name="decoder_pipeline")
            logger.info("Артефакты успешно синхронизированы в %s", cache_base)
        except Exception as e:
            logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов Decoder: %s", e)
            raise RuntimeError("Artifact resolution failed.") from e

        # 2. Подключение к vLLM
        logger.info("Подключение к внешнему LLM-серверу...")
        llm_url = cfg.services.get("llm_api_url", "http://localhost:8000/v1")
        generator = LLMGenerationClient(
            api_base=llm_url,
            temperature=cfg.api.get("generation", {}).get("temperature", 0.7),
        )
        app.state.ml_models["generator"] = generator

        # 3. Telegram-бот
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
