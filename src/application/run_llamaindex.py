# scripts/application/run_llamaindex.py
"""Демо-скрипт: LlamaIndex как альтернативный оркестратор RAG-пайплайна.

Использует те же модели (энкодер + декодер), что и основной RAGOrchestrator,
но оркестрацию (ретривал → промпт → генерация) делегирует LlamaIndex.

Загружает существующий FAISS-индекс с диска (созданный через index_db.py)
вместо создания нового пустого индекса.
"""

import logging
from pathlib import Path

import faiss
import hydra
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from omegaconf import DictConfig

from src.application.llamaindex_ext import DecoderPipelineLLM, RAGPipelineEmbedding
from src.pipelines.decoder.inference.inference import LLMGenerationPipeline
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("Инициализация LlamaIndex-пайплайна...")

    # ── 1. LLM (Декодер) ─────────────────────────────────────────────────────
    logger.info("Загрузка LLMGenerationPipeline...")
    llm_pipeline = LLMGenerationPipeline(config_name="main")
    Settings.llm = DecoderPipelineLLM(generator=llm_pipeline)

    # ── 2. Эмбеддер (RAG-энкодер) ────────────────────────────────────────────
    logger.info("Загрузка RAGInferenceEmbedder...")
    encoder_tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
    encoder_builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    encoder_model = encoder_builder.build(tokenizer=encoder_tokenizer)
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    embedder = hydra.utils.instantiate(
        cfg.rag_pipeline.inference.embedder,
        model=encoder_model,
        pooler=pooler,
        tokenizer=encoder_tokenizer,
    )
    Settings.embed_model = RAGPipelineEmbedding(embedder=embedder)

    # ── 3. FAISS-индекс с диска ───────────────────────────────────────────────
    # Загружаем индекс, созданный через index_db.py, а не создаём пустой.
    # FAISSVectorDB хранит метаданные отдельно — для LlamaIndex они не нужны,
    # нам нужен только сам faiss.Index объект.
    db_dir = Path(cfg.paths.db_dir)
    index_path = db_dir / "index.faiss"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS-индекс не найден: {index_path}. Сначала запустите scripts/rag/index_db.py."
        )

    logger.info("Загрузка FAISS-индекса из '%s'...", index_path)
    raw_faiss_index = faiss.read_index(str(index_path))
    logger.info("Индекс загружен: %d документов.", raw_faiss_index.ntotal)

    vector_store = FaissVectorStore(faiss_index=raw_faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── 4. VectorStoreIndex поверх загруженного индекса ──────────────────────
    # from_vector_store — не переиндексирует документы, использует готовый индекс.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    # ── 5. Query Engine и тестовый запрос ────────────────────────────────────
    top_k = cfg.rag_pipeline.inference.get("top_k", 3)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    query = cfg.rag_pipeline.inference.get(
        "test_query",
        "Какие существуют архитектурные паттерны для масштабирования RAG?",
    )
    logger.info("Запрос: '%s'", query)

    response = query_engine.query(query)

    logger.info("=== Ответ LLM ===")
    logger.info("%s", response.response)

    logger.info("=== Использованный контекст (top_%d) ===", top_k)
    for node in response.source_nodes:
        logger.info("- [score=%.4f] %s", node.score, node.text[:120])


if __name__ == "__main__":
    main()
