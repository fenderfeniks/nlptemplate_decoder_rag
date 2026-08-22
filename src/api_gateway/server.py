# src/api_gateway/server.py
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.api_gateway.endpoints import chat
from src.api_gateway.middlewares import setup_gateway_middlewares
from src.api_gateway.telemetry import setup_telemetry
from src.application.orchestrator import RAGOrchestrator
from src.pipelines.decoder.core.prompts.manager import PromptManager
from src.pipelines.decoder.inference.inference import LLMGenerationClient

logger = logging.getLogger(__name__)


def create_gateway_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Инициализация API Gateway...")

        storage_root = Path(os.getenv("STORAGE_ROOT", "prod_storage"))
        manifest = json.loads((storage_root / "manifest.json").read_text(encoding="utf-8"))
        model_name = manifest["decoder_pipeline"]["mlflow_model_name"]

        llm_client = LLMGenerationClient(
            api_base=os.getenv("LLM_API_URL", "http://localhost:8081/v1"),
            model_name=model_name,
        )

        rag_qa_path = Path("configs/evaluation/prompts/rag_qa.yaml")
        rag_qa_config = yaml.safe_load(rag_qa_path.read_text(encoding="utf-8"))
        prompt_manager = PromptManager(templates=rag_qa_config["templates"])

        orchestrator = RAGOrchestrator(
            rag_api_url=os.getenv("RAG_API_URL", "http://localhost:8001"),
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            http_timeout=float(os.getenv("HTTP_TIMEOUT", "10.0")),
        )

        app.state.orchestrator = orchestrator
        logger.info("API Gateway готов. model=%s", model_name)

        yield

        await orchestrator.close()
        logger.info("API Gateway остановлен.")

    app = FastAPI(
        title="NLP API Gateway",
        description="Единая точка входа для RAG и LLM",
        lifespan=lifespan,
    )

    # Telemetry инициализируется до middleware и роутеров.
    # FastAPIInstrumentor внутри setup_telemetry добавляет свой middleware —
    # он должен быть зарегистрирован раньше пользовательских middleware,
    # иначе входящий trace-context не будет прочитан до нашей обработки.
    setup_telemetry(app, service_name=os.getenv("OTEL_SERVICE_NAME", "api-gateway"))

    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    setup_gateway_middlewares(app, cors_origins)

    app.include_router(chat.router)

    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

    return app