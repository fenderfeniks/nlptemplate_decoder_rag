# src/api_gateway/server.py
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.api_gateway.endpoints import chat
from src.api_gateway.middlewares import setup_gateway_middlewares
from src.pipelines.decoder.core.prompts.manager import PromptManager
from src.pipelines.decoder.inference.inference import LLMGenerationClient


logger = logging.getLogger(__name__)


def create_gateway_app() -> FastAPI:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="main")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Инициализация API Gateway...")

        # 1. Менеджер промптов
        prompt_manager = PromptManager(templates=cfg.get("prompts", {}))

        # 2. Клиент к vLLM — URL берётся из Hydra-конфига
        llm_api_url = os.getenv("LLM_API_URL", "http://localhost:8000/v1")
        llm_client = LLMGenerationClient(api_base=llm_api_url)

        # 3. Оркестратор — все параметры из конфига
        rag_api_url = os.getenv("RAG_API_URL", "http://localhost:8001")
        orchestrator = hydra.utils.instantiate(
            cfg.application.orchestrator,
            rag_api_url=rag_api_url,
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            http_timeout=cfg.get("http_timeout", 10.0),
        )
        app.state.orchestrator = orchestrator
        logger.info("API Gateway готов принимать запросы.")

        yield

        logger.info("Остановка API Gateway...")
        await orchestrator.close()

    app = FastAPI(
        title="NLP API Gateway",
        description="Единая точка входа для RAG и LLM",
        lifespan=lifespan,
    )

    # Middleware (CORS + логирование + метрики)
    cors_origins: list[str] = list(cfg.get("cors_origins", ["*"]))
    setup_gateway_middlewares(app, cors_origins)

    app.include_router(chat.router)

    return app
