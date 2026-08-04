# src/rag_pipeline/api/server.py
import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.pipelines.rag.api.rest.endpoints import health, search
from src.pipelines.rag.api.rest.middlewares import setup_middlewares
from src.rag_pipeline.api.rest.limiter import limiter
from src.vector_store.vector_db import FAISSVectorDB


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

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
    """Фабрика FastAPI-приложения.

    Загружает Hydra-конфиг из ``configs/`` относительно корня проекта,
    инициализирует RAG-стек в lifespan и регистрирует роутеры.
    """
    load_dotenv()
    config_dir = Path(__file__).resolve().parents[4] / "configs"

    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="main")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ml_models = {}
        logger.info("Загрузка RAG-стека в память...")

        try:
            # 1. Токенизатор и энкодер
            tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
            builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
            base_model = builder.build(tokenizer=tokenizer)
            pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

            # 2. Эмбеддер
            embedder = hydra.utils.instantiate(
                cfg.rag_pipeline.inference,
                model=base_model,
                pooler=pooler,
                tokenizer=tokenizer,
            )

            # 3. Загрузка FAISS-индекса с диска
            # Параметры должны совпадать с теми, что использовались при index_db.py
            db_dir = Path(cfg.paths.db_dir)
            vector_db_cfg = hydra.utils.instantiate(cfg.vector_db)
            vector_db = FAISSVectorDB.load(
                directory=db_dir,
                embedding_dim=vector_db_cfg.embedding_dim,
                index_type=vector_db_cfg.index_type,
                normalize_embeddings=vector_db_cfg.normalize_embeddings,
            )
            logger.info(
                "FAISS-индекс загружен из '%s' (%d документов).",
                db_dir,
                vector_db.index.ntotal,
            )

            # 4. Ретривер
            retriever = hydra.utils.instantiate(
                cfg.rag_pipeline.retrieval,
                embedder=embedder,
                vector_db=vector_db,
            )

            app.state.ml_models["retriever"] = retriever
            logger.info("RAG-стек успешно запущен и готов к работе.")

        except Exception:
            logger.exception("Критическая ошибка при старте RAG API:")
            raise

        yield

        # Shutdown: освобождаем память
        app.state.ml_models.clear()
        gc.collect()
        logger.info("RAG API: ресурсы освобождены.")

    # CORS origins из конфига или env — не хардкодим "*" в проде
    cors_origins_raw = os.getenv(
        "CORS_ORIGINS",
        OmegaConf.select(cfg, "rag_pipeline.api.cors_origins", default="*"),
    )
    cors_origins = (
        [o.strip() for o in cors_origins_raw.split(",")]
        if isinstance(cors_origins_raw, str)
        else list(cors_origins_raw)
    )

    app = FastAPI(
        title="RAG Retrieval API",
        description="Векторный поиск по базе знаний на основе FAISS.",
        version=os.getenv("APP_VERSION", "0.1.0"),
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    setup_middlewares(app, cors_origins=cors_origins)
    app.include_router(health.router)
    app.include_router(search.router)
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

    return app


app = create_app()
