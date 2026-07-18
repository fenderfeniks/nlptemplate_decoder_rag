"""
Job для Airflow: Парсинг, очистка и загрузка документов в векторную БД.
"""

import logging
import os
import re

from dotenv import load_dotenv


# Загружаем секреты до старта Гидры
load_dotenv()

import hydra  # noqa: E402
from llama_index.core import Document, SimpleDirectoryReader  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.core.rag.indexer import RAGIndexer  # noqa: E402


logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Индустриальная очистка сырого текста перед векторизацией."""
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"http[s]?://\S+", "[ССЫЛКА]", text)

    return text.strip()


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logger.info("Запуск ETL Pipeline для базы знаний RAG...")

    # ИСПРАВЛЕНИЕ: Безопасное склеивание путей, независимое от ОС
    raw_docs_dir = os.path.join(cfg.paths.data_dir, "raw", "knowledge_base")
    logger.info(f"Чтение сырых документов из: {raw_docs_dir}")

    raw_documents = SimpleDirectoryReader(raw_docs_dir).load_data()

    logger.info("Очистка документов от мусора...")
    cleaned_documents = []
    for doc in raw_documents:
        cleaned_text = clean_text(doc.text)
        cleaned_documents.append(Document(text=cleaned_text, metadata=doc.metadata))

    logger.info("Инициализация RAG Indexer...")
    indexer: RAGIndexer = hydra.utils.instantiate(cfg.rag.indexer)

    # ИСПРАВЛЕНИЕ: Актуализируем текст лога в соответствии с архитектурой FAISS
    logger.info("Нарезка на чанки, векторизация и сохранение в локальный индекс FAISS...")
    indexer.build_and_save_index(documents=cleaned_documents)

    logger.info("ETL Pipeline успешно завершен! База знаний обновлена.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()