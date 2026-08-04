# src/pipelines/rag/api/server.py
# Показан только изменённый фрагмент _load_rag_stack_sync —
# остальной код server.py остаётся без изменений.
#
# Было:
#   from src.vector_store.faiss_store import FAISSVectorStore
#   vector_db = FAISSVectorStore.load(directory=db_dir, ...)
#
# Стало:
#   vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)
#
# При смене бэкенда меняем только configs/vector_db/*.yaml — server.py не трогаем.

import asyncio
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
from src.pipelines.rag.api.rest.limiter import limiter
from src.pipelines.rag.api.rest.middlewares import setup_middlewares
from src.tools.storage.resolver import ArtifactResolver


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


_MODEL_LOAD_TIMEOUT_SEC: int = int(os.getenv("MODEL_LOAD_TIMEOUT_SEC", "120"))


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key or ""


def _load_rag_stack_sync(cfg: object, app: FastAPI) -> None:
    """Синхронная загрузка RAG-стека."""

    # 1. Резолвинг артефактов (Скачивание + Патчинг конфигов)
    router = hydra.utils.instantiate(cfg.storage_router)
    resolver = ArtifactResolver(
        router=router, cache_base_dir=Path(cfg.paths.model_dir) / "rag_cache"
    )

    manifest_uri = os.getenv("MANIFEST_URI", "local://./prod_storage/manifests/rag_manifest.json")

    try:
        # Эта одна строчка делает всё: качает манифест, веса, БД и патчит cfg
        db_dir = resolver.resolve_and_patch(cfg, manifest_uri, pipeline_name="rag_pipeline")
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов RAG: %s", e)
        raise RuntimeError("Artifact resolution failed.") from e

    # 2. Сборка токенизатора и энкодера (уже с пропатченными локальными путями)
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    # 3. Эмбеддер
    embedder = hydra.utils.instantiate(
        cfg.rag_pipeline.inference,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # 4. Векторное хранилище
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)

    logger.info("Векторное хранилище загружено из '%s' (%d документов).", db_dir, vector_db.ntotal)

    # 5. Ретривер
    retriever = hydra.utils.instantiate(
        cfg.rag_pipeline.retrieval,
        embedder=embedder,
        vector_db=vector_db,
    )

    app.state.ml_models["retriever"] = retriever
    logger.info("RAG-стек успешно запущен.")


def create_app() -> FastAPI:
    load_dotenv()
    config_dir = Path(__file__).resolve().parents[4] / "configs"

    if not os.getenv("API_KEY"):
        logger.warning(
            "API_KEY не задан — все запросы к защищённым эндпоинтам "
            "пропускаются без проверки ключа."
        )

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
        logger.info("Загрузка RAG-стека (таймаут=%ds)...", _MODEL_LOAD_TIMEOUT_SEC)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(_load_rag_stack_sync, cfg, app),
                timeout=_MODEL_LOAD_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.critical(
                "Загрузка RAG-стека превысила таймаут %ds — завершаем процесс.",
                _MODEL_LOAD_TIMEOUT_SEC,
            )
            raise RuntimeError(f"Model load timeout after {_MODEL_LOAD_TIMEOUT_SEC}s") from None
        except Exception:
            logger.exception("Критическая ошибка при старте RAG API:")
            raise

        yield

        app.state.ml_models.clear()
        gc.collect()
        logger.info("RAG API: ресурсы освобождены.")

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
        description="Векторный поиск по базе знаний.",
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
