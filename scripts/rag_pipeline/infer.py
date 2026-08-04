import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("Инициализация RAG-ретривера...")

    # 1. Резолвинг артефактов (Энкодер + БД)
    router = hydra.utils.instantiate(cfg.storage_router)
    cache_base = Path(cfg.paths.model_dir) / "rag_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    manifest_uri = cfg.get("manifest_uri", "local://./prod_storage/manifests/rag_manifest.json")

    try:
        db_dir = resolver.resolve_and_patch(cfg, manifest_uri, pipeline_name="rag_pipeline")
        if not db_dir:
            raise ValueError("Манифест не содержит 'vector_db_uri'. База не найдена.")
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов RAG: %s", e)
        sys.exit(1)

    # 2. Сборка Энкодера (с уже пропатченными локальными путями)
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    embedder = hydra.utils.instantiate(
        cfg.rag_pipeline.inference,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # 3. Динамическая сборка Векторной БД
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)
    logger.info("Векторная БД загружена из '%s' (%d документов).", db_dir, vector_db.ntotal)

    # 4. Сборка ретривера
    retriever = hydra.utils.instantiate(
        cfg.rag_pipeline.retrieval,
        embedder=embedder,
        vector_db=vector_db,
    )

    # 5. Тестовый прогон
    query: str = cfg.rag_pipeline.inference.get("test_query", "Тестовый запрос")
    top_k: int = cfg.rag_pipeline.inference.get("top_k", 3)

    logger.info("Запрос: '%s'", query)
    results = retriever.search(query, top_k=top_k)

    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        text = res.get("metadata", {}).get("text", "").replace("\n", " ")
        logger.info("[%d] score=%.4f | текст: %s...", i, score, text[:150])


if __name__ == "__main__":
    infer()
