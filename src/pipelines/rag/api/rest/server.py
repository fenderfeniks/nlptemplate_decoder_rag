# src/pipelines/rag/api/server.py


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


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key or ""


def _load_rag_stack_sync(cfg: object, app: FastAPI) -> None:
    """Синхронная загрузка RAG-стека."""

    # 1. Резолвинг артефактов (Скачивание + Патчинг конфигов)
    router = hydra.utils.instantiate(cfg.system.storage_router)
    resolver = ArtifactResolver(
        router=router, cache_base_dir=Path(cfg.system.paths.model_dir) / "rag_cache"
    )
    manifest_uri = cfg.system.manifest.uri

    db_dir, lora_path, _ = resolver.resolve_and_patch(
        cfg, manifest_uri, pipeline_name="rag_pipeline", is_training=False
    )
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    OmegaConf.update(cfg, "model.builder.modifiers", None, force_add=True)
    builder = hydra.utils.instantiate(cfg.model.builder)
    base_model = builder.build(tokenizer=tokenizer)

    if lora_path:
        from peft import PeftModel

        base_model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=False)

    pooler = hydra.utils.instantiate(cfg.model.pooling)

    # 3. Эмбеддер — фильтруем конфиг через сигнатуру __init__,
    from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder


    _emb = cfg.inference.embedder  # вот настоящий конфиг embedder'а
    embedder = RAGInferenceEmbedder(
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
        device=_emb.get("device", "cuda"),
        precision=_emb.get("precision", "bf16"),
        max_length=_emb.get("max_length", 512),
    )
    logger.info("embedder создан: %s", type(embedder))

    # 4. Векторное хранилище
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)

    logger.info("Векторное хранилище загружено из '%s' (%d документов).", db_dir, vector_db.ntotal)

    reranker = None
    reranker_cfg = OmegaConf.select(cfg, "rag_pipeline.reranker", default=None)

    if reranker_cfg is not None:
        try:
            _, reranker_lora, _ = resolver.resolve_and_patch(
                cfg,
                cfg.system.manifest.uri,
                pipeline_name="reranker_pipeline",
                is_training=False,
            )
            OmegaConf.update(
                cfg,
                "model.builder.auto_model_class",
                "transformers.AutoModelForSequenceClassification",
            )
            reranker_tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
            reranker_builder = hydra.utils.instantiate(cfg.model.builder)
            reranker_model = reranker_builder.build(tokenizer=reranker_tokenizer)
            if reranker_lora:
                from peft import PeftModel
                reranker_model = PeftModel.from_pretrained(reranker_model, str(reranker_lora), is_trainable=False)
            reranker = hydra.utils.instantiate(
                reranker_cfg,
                model=reranker_model,
                tokenizer=reranker_tokenizer,
            )
            logger.info("CrossEncoderReranker инициализирован.")
        except Exception as e:
            logger.error("ОШИБКА инициализации реранкера: %s. Работаем без реранкинга.", e)
            reranker = None
    else:
        logger.info("Реранкер не задан в конфиге — работаем без реранкинга.")

    # --- Диагностика перед созданием ретривера ---
    logger.info("embedder type: %s", type(embedder))
    logger.info("embedder is RAGInferenceEmbedder: %s", isinstance(embedder, RAGInferenceEmbedder))

    # Проверяем конфиг ретривера на наличие вложенного embedder-а,
    # который Hydra может предпочесть переданному keyword-аргументу.
    _retriever_cfg = cfg.inference.retriever
    logger.info(
        "cfg.inference.retriever keys: %s",
        list(_retriever_cfg.keys()) if hasattr(_retriever_cfg, "keys") else _retriever_cfg,
    )
    _has_embedded_embedder_cfg = (
        hasattr(_retriever_cfg, "keys") and "embedder" in _retriever_cfg
    )
    if _has_embedded_embedder_cfg:
        logger.warning(
            "cfg.inference.retriever содержит ключ 'embedder' (%s). "
            "Hydra может подставить DictConfig вместо переданного инстанса. "
            "Удаляем из конфига — embedder передаётся напрямую.",
            type(_retriever_cfg.embedder),
        )
        # Создаём копию конфига без ключа embedder, чтобы Hydra
        # не перезаписала переданный инстанс своим DictConfig.
        from omegaconf import OmegaConf as _OmegaConf2
        _retriever_cfg = _OmegaConf2.masked_copy(
            _retriever_cfg,
            [k for k in _retriever_cfg if k != "embedder"],
        )

    # --- 5б. Ретривер (с реранкером) ---
    retriever = hydra.utils.instantiate(
        _retriever_cfg,
        embedder=embedder,
        vector_db=vector_db,
        reranker=reranker,          # None -> реранкинг отключён
        # rerank_factor задаётся в retrieval.yaml, по умолчанием 3
    )
    logger.info("retriever.embedder type after instantiate: %s", type(getattr(retriever, "embedder", None)))
    
    app.state.ml_models["retriever"] = retriever
    logger.info("RAG-стек успешно запущен%s.", " (с реранкером)" if reranker else "")


def create_app() -> FastAPI:
    load_dotenv()
    config_dir = Path(__file__).resolve().parents[5] / "configs"

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
        cfg = hydra.compose(config_name="rag_api")
        OmegaConf.resolve(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = cfg.api.server.model_load_timeout_sec
        app.state.ml_models = {}
        logger.info("Загрузка RAG-стека (таймаут=%ds)...", timeout)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(_load_rag_stack_sync, cfg, app),
                timeout = timeout,
            )
        except asyncio.TimeoutError:
            logger.critical(
                "Загрузка RAG-стека превысила таймаут %ds — завершаем процесс.",
                timeout,
            )
            raise RuntimeError(f"Model load timeout after {timeout}s") from None
        except Exception:
            logger.exception("Критическая ошибка при старте RAG API:")
            raise

        yield

        app.state.ml_models.clear()
        gc.collect()
        logger.info("RAG API: ресурсы освобождены.")

    raw = cfg.api.server.cors_origins
    if isinstance(raw, str):
        cors_origins = [o.strip() for o in raw.split(",")]
    else:
        cors_origins = list(raw)

    app = FastAPI(
        title=cfg.api.server.title,
        version=cfg.api.server.version,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.state.rate_limit = cfg.api.rate_limit.default_limit
    app.state.service_name = cfg.api.service_name
    limiter._enabled = cfg.api.rate_limit.enabled
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