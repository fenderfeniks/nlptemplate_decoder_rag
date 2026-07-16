"""
Job для Airflow: Парсинг, очистка и загрузка документов в векторную БД.
Запускается по расписанию (например, каждую ночь).
"""

import re
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader

# Импортируем наш индексатор, который мы писали ранее
from src.core.rag.indexer import RAGIndexer

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Индустриальная очистка сырого текста перед векторизацией.
    Удаляет мусор, который портит качество эмбеддингов.
    """
    if not text:
        return ""
    
    # 1. Удаляем HTML теги (если парсили страницы интранета)
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Удаляем множественные переносы строк и пробелы
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # 3. Удаляем URL-ссылки (опционально, если они не нужны модели)
    text = re.sub(r'http[s]?://\S+', '[ССЫЛКА]', text)
    
    return text.strip()


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logger.info("Запуск ETL Pipeline для базы знаний RAG...")

    # В реальном Enterprise здесь был бы код выгрузки файлов из AWS S3 или Confluence API 
    # во временную локальную папку. Для шаблона берем файлы из директории data/.
    raw_docs_dir = cfg.paths.data_dir + "/raw/knowledge_base"
    
    logger.info(f"Чтение сырых документов из: {raw_docs_dir}")
    raw_documents = SimpleDirectoryReader(raw_docs_dir).load_data()
    
    logger.info("Очистка документов от мусора...")
    cleaned_documents = []
    for doc in raw_documents:
        cleaned_text = clean_text(doc.text)
        # Создаем новый объект Document из LlamaIndex с чистым текстом и сохраняем метаданные
        cleaned_documents.append(Document(text=cleaned_text, metadata=doc.metadata))

    logger.info("Инициализация RAG Indexer...")
    # Инициализируем наш класс (который мы написали ранее)
    indexer = hydra.utils.instantiate(cfg.rag.indexer)
    
    # Передаем ему уже ОЧИЩЕННЫЕ документы
    logger.info("Нарезка на чанки, векторизация и отправка в Qdrant...")
    indexer.build_and_save_index(documents=cleaned_documents)
    
    logger.info("ETL Pipeline успешно завершен! База знаний обновлена.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()