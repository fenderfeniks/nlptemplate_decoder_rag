# scripts/rag/infer.py
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.pipelines.rag.inference.builder import build_inference_encoder
from src.pipelines.rag.inference.embedder_factory import build_embedder
from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    """Тестовый прогон RAG-ретривера по одному запросу."""
    cfg = setup_config(cfg)
    logger.info("Инициализация RAG-ретривера...")

    # 1. Резолвинг артефактов (энкодер + БД)
    router = hydra.utils.instantiate(cfg.storage_router)
    cache_base = Path(cfg.paths.model_dir) / "rag_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    try:
        db_dir, lora_path = resolver.resolve_and_patch(
            cfg, cfg.manifest.uri, pipeline_name="rag_pipeline"
        )
        if not db_dir:
            raise ValueError("Манифест не содержит 'vector_db_uri'. База не найдена.")
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов RAG: %s", e)
        sys.exit(1)

    # 2. Сборка энкодера и эмбеддера
    base_model, pooler, tokenizer = build_inference_encoder(cfg, lora_path)
    embedder = build_embedder(cfg, base_model, pooler, tokenizer)

    # 3. Загрузка векторной БД
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)
    logger.info("Векторная БД загружена из '%s' (%d документов).", db_dir, vector_db.ntotal)

    # 4. Сборка ретривера и тестовый запрос
    # test_query / top_k — поля inference.yaml для smoke-теста, не параметры эмбеддера.
    inference_cfg = cfg.rag_pipeline.inference
    query: str = inference_cfg.get("test_query", "Тестовый запрос")
    top_k: int = inference_cfg.get("top_k", 3)

    retriever = hydra.utils.instantiate(
        cfg.rag_pipeline.retrieval,
        embedder=embedder,
        vector_db=vector_db,
    )

    logger.info("Запрос: '%s'", query)
    results = retriever.search(query, top_k=top_k)

    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        text = res.get("metadata", {}).get("text", "").replace("\n", " ")
        logger.info("[%d] score=%.4f | текст: %s...", i, score, text[:150])


if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline

    enforce_pipeline("rag_pipeline")
    infer()
