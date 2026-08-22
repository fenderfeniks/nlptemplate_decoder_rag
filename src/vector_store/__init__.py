# src/vector_store/__init__.py
"""Векторное хранилище с подключаемыми бэкендами.

Публичный API::

    from src.vector_store import BaseVectorStore, FAISSVectorStore, QdrantVectorStore, LSHIndex

Смена бэкенда — только в конфиге (main.yaml)::

    # было:
    - vector_db: flat
    # стало:
    - vector_db: qdrant

Оба реализуют ``BaseVectorStore`` — остальной код не меняется.
"""

from src.vector_store.base import BaseVectorStore

# from src.vector_store.faiss_store import FAISSVectorStore
from src.vector_store.lsh import LSHIndex


# QdrantVectorStore — опциональная зависимость.
# Импортируется только если qdrant-client установлен.
try:
    from src.vector_store.qdrant_store import QdrantVectorStore

    __all__ = [
        "BaseVectorStore",
        "FAISSVectorStore",
        "QdrantVectorStore",
        "LSHIndex",
    ]
except ImportError:
    __all__ = [
        "BaseVectorStore",
        "FAISSVectorStore",
        "LSHIndex",
    ]
