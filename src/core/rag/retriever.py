"""
Модуль извлечения (Retrieval).
Загружает готовую векторную БД и достает релевантные куски текста по запросу.
"""

import logging
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(
        self,
        persist_dir: str,
        similarity_top_k: int = 3,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_dir = persist_dir
        self.similarity_top_k = similarity_top_k
        
        # Ретриверу нужна та же модель эмбеддингов, чтобы понять вопрос!
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.llm = None 
        
        logger.info(f"Загрузка векторной БД из {self.persist_dir}...")
        # Загружаем пустой FAISS (он сам подхватит веса с диска)
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            persist_dir=persist_dir
        )
        
        self.index = load_index_from_storage(storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=3)

    def retrieve_context(self, query: str) -> str:
        """
        Ищет ответ в базе и возвращает склеенный текст (контекст).
        """
        logger.info(f"Поиск в базе по запросу: '{query}'")
        nodes = self.retriever.retrieve(query)
        
        # Достаем текст из найденных "нод" (кусков) и склеиваем через перенос строки
        context = "\n\n".join([node.get_content() for node in nodes])
        return context