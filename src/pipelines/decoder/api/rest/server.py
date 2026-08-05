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


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Проверяет API-ключ из заголовка X-API-Key."""
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key or ""


def create_app() -> FastAPI:
    load_dotenv()

    if not os.getenv("API_KEY"):
        logger.warning(
            "API_KEY не задан — все запросы к защищённым эндпоинтам "
            "пропускаются без проверки ключа."
        )

    config_dir = Path(__file__).resolve().parents[5] / "configs"

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

        # Подключение к стандартизированному vLLM серверу
        logger.info("Инициализация клиента LLM...")
        llm_url = os.getenv("LLM_API_URL", "http://localhost:8000/v1")

        generator = LLMGenerationClient(
            api_base=llm_url,
            temperature=cfg.decoder_pipeline.api.get("generation", {}).get("temperature", 0.7),
        )
        app.state.ml_models["generator"] = generator
        logger.info("Клиент LLM успешно настроен на адрес: %s", llm_url)

        yield

        app.state.ml_models.clear()

    app = FastAPI(
        title=cfg.decoder_pipeline.api.title,
        description=cfg.decoder_pipeline.api.description,
        version=cfg.decoder_pipeline.api.version,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.state.config = cfg

    setup_middlewares(app, cors_origins=list(cfg.decoder_pipeline.api.cors_origins))

    app.include_router(health.router)
    app.include_router(generate.router, dependencies=[Depends(verify_api_key)])

    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

    return app


app = create_app()
