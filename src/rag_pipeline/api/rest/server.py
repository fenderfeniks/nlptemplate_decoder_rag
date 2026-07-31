# src/rag_pipeline/api/server.py
import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
from fastapi import FastAPI
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.rag_pipeline.api.rest.endpoints import health, search
from src.rag_pipeline.api.rest.limiter import limiter
from src.rag_pipeline.api.rest.middlewares import setup_middlewares
from src.utils.vector_db import FAISSVectorDB


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Фабрика FastAPI-приложения.

    Загружает Hydra-конфиг из ``configs/`` относительно корня проекта,
    инициализирует RAG-стек в lifespan и регистрирует роутеры.
    """
    config_dir = Path(__file__).resolve().parents[4] / "configs"
    GlobalHydra.instance().clear()

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

    return app


app = create_app()
