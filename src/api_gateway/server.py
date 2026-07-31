import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.api_gateway.endpoints import chat
from src.application.orchestrator import RAGOrchestrator
from src.decoder_pipeline.core.prompts.manager import PromptManager
from src.decoder_pipeline.sdk.inference import LLMGenerationClient


logger = logging.getLogger(__name__)


def create_gateway_app() -> FastAPI:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="main")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Инициализация API Gateway...")

        # 1. Менеджер промптов (работает в памяти, шаблоны из конфига)
        prompt_manager = PromptManager(templates=cfg.get("prompts", {}))

        # 2. Клиент к vLLM (генерация)
        llm_url = os.getenv("LLM_API_URL", "http://llm-service:8000/v1")
        llm_client = LLMGenerationClient(api_base=llm_url)

        # 3. Оркестратор (связывает RAG API и LLM API)
        rag_url = os.getenv("RAG_API_URL", "http://rag-api:8001")
        orchestrator = RAGOrchestrator(
            rag_api_url=rag_url,
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            default_template="rag_qa",
            default_top_k=cfg.get("top_k", 5),
        )

        app.state.orchestrator = orchestrator
        logger.info("API Gateway готов принимать запросы.")

        yield

        # Корректное закрытие асинхронных HTTP-сессий при выключении
        logger.info("Остановка API Gateway...")
        await orchestrator.close()

    app = FastAPI(
        title="NLP API Gateway", description="Единая точка входа для RAG и LLM", lifespan=lifespan
    )

    app.include_router(chat.router)

    return app
