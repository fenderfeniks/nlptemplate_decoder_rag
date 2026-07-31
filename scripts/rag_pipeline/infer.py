# scripts/rag/infer.py
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging
from src.utils.vector_db import FAISSVectorDB


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    """Тестовый прогон онлайн-ретривера с выводом результатов в лог.

    Загружает сохранённый FAISS-индекс из ``cfg.paths.db_dir``.
    Тестовый запрос задаётся через ``cfg.rag_pipeline.inference.test_query``
    или хардкодится как дефолт.
    """
    cfg = setup_config(cfg)
    logger.info("Инициализация онлайн-ретривера...")

    # 1. Загрузка токенизатора и энкодера
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    # 2. Сборка эмбеддера
    embedder = hydra.utils.instantiate(
        cfg.rag_pipeline.inference,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # 3. Загрузка сохранённого FAISS-индекса с диска
    db_dir = Path(cfg.paths.db_dir)
    vector_db_cfg = hydra.utils.instantiate(cfg.vector_db)
    vector_db = FAISSVectorDB.load(
        directory=db_dir,
        embedding_dim=vector_db_cfg.embedding_dim,
        index_type=vector_db_cfg.index_type,
        normalize_embeddings=vector_db_cfg.normalize_embeddings,
    )
    logger.info("FAISS-индекс загружен из '%s' (%d документов).", db_dir, vector_db.index.ntotal)

    # 4. Сборка ретривера
    retriever = hydra.utils.instantiate(
        cfg.rag_pipeline.retrieval,
        embedder=embedder,
        vector_db=vector_db,
    )

    # 5. Тестовый прогон
    query: str = cfg.rag_pipeline.inference.get(
        "test_query",
        "Какие существуют архитектурные паттерны для масштабирования RAG?",
    )
    top_k: int = cfg.rag_pipeline.inference.get("top_k", 3)

    logger.info("Запрос: '%s'", query)
    results = retriever.search(query, top_k=top_k)

    logger.info("--- Результаты поиска (top_%d) ---", top_k)
    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        meta = res.get("metadata", {})
        doc_id = meta.get("doc_id", "unknown")
        text = meta.get("text", "").replace("\n", " ")
        logger.info("[%d] score=%.4f | id=%s | текст: %s...", i, score, doc_id, text)


if __name__ == "__main__":
    infer()
