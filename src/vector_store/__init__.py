# src/vector_store/__init__.py
"""Векторное хранилище с подключаемыми бэкендами.

Публичный API::

    from src.vector_store import BaseVectorStore, FAISSVectorStore, LSHIndex

Смена бэкенда — только в конфиге::

    # было:
    store = FAISSVectorStore(embedding_dim=768)
    # стало:
    store = QdrantVectorStore(embedding_dim=768, url="http://localhost:6333")

Оба реализуют ``BaseVectorStore`` — весь остальной код не меняется.
"""

from src.vector_store.base import BaseVectorStore
from src.vector_store.faiss_store import FAISSVectorStore
from src.vector_store.lsh import LSHIndex


__all__ = [
    "BaseVectorStore",
    "FAISSVectorStore",
    "LSHIndex",
]
