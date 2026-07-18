import logging
import os

import faiss
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


logger = logging.getLogger(__name__)


class RAGIndexer:
    def __init__(
        self,
        documents_dir: str,
        persist_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dimension: int = 384,
        hnsw_m: int = 32,
        device: str = "cpu",  # ИСПРАВЛЕНИЕ: Выносим эмбеддер на CPU по умолчанию
    ):
        self.documents_dir = documents_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_dimension = vector_dimension
        self.hnsw_m = hnsw_m
        self.device = device

        logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name} на устройстве: {self.device}")
        # ИСПРАВЛЕНИЕ: Явно передаем device, чтобы не отъедать VRAM у основной LLM
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name, 
            device=self.device
        )
        Settings.llm = None
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def build_and_save_index(self, documents: list[Document] | None = None):
        """
        Создает индекс. Если documents не передан, читает сырые файлы с диска.
        """
        if documents is None:
            logger.info(f"Чтение сырых документов из: {self.documents_dir}")
            if not os.path.exists(self.documents_dir):
                raise FileNotFoundError(f"Папка не найдена: {self.documents_dir}")
            documents = SimpleDirectoryReader(self.documents_dir).load_data()
        else:
            logger.info(f"Получено {len(documents)} очищенных документов из пайплайна.")

        faiss_index = faiss.IndexHNSWFlat(self.vector_dimension, self.hnsw_m)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info("Создание HNSW индекса...")
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("Индекс успешно сохранен.")