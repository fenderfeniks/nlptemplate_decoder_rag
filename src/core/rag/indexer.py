"""
Модуль для создания векторного индекса (Data Ingestion).
Берет сырые документы, чанкует, векторизует и сохраняет в БД.
"""

import faiss
import os
import logging
from typing import Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# В индустрии часто используют эмбеддинги от HuggingFace
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class RAGIndexer:
    def __init__(
        self,
        documents_dir: str,
        persist_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.documents_dir = documents_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Настраиваем глобальные параметры LlamaIndex
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        
        # Отключаем встроенную LLM от LlamaIndex, так как мы будем генерировать 
        # ответы сами через нашу кастомную модель в generation.py
        Settings.llm = None 
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def build_and_save_index(self):
        documents = SimpleDirectoryReader(self.documents_dir).load_data()
        
        # 1. Настройка FAISS с HNSW
        # Наша модель all-MiniLM-L6-v2 выдает векторы размером 384
        vector_dimension = 384  
        
        # M = 32 — это гиперпараметр HNSW (количество связей в графе). 
        # Чем больше, тем точнее поиск, но больше жрет оперативки. В индустрии берут 16-64.
        faiss_index = faiss.IndexHNSWFlat(vector_dimension, 32)
        
        # 2. Оборачиваем FAISS в формат, понятный LlamaIndex
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # 3. Создаем "Контекст хранилища"
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        logger.info("Создание HNSW индекса...")
        # 4. Передаем контекст при создании!
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )
        
        # Сохраняем граф FAISS на диск
        index.storage_context.persist(persist_dir=self.persist_dir)