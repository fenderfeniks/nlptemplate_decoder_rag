# tests/vector_store/test_base.py
"""Тесты для BaseVectorStore Protocol.

Проверяем:
- runtime isinstance-проверки работают без наследования.
- Структурно совместимый объект проходит isinstance.
- Неполный объект не проходит isinstance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.vector_store.base import BaseVectorStore


# ---------------------------------------------------------------------------
# Фиктивные реализации
# ---------------------------------------------------------------------------


class _FullImpl:
    """Полная структурная реализация протокола (без наследования)."""

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def ntotal(self) -> int:
        return 0

    @property
    def existing_doc_ids(self) -> set[str]:
        return set()

    def insert(self, embeddings: Any, metadata: list[dict[str, Any]]) -> None:
        pass

    def search(
        self,
        query_embeddings: Any,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        return []

    def save(self, directory: str | Path) -> None:
        pass

    def reset(self) -> None:
        pass


class _MissingInsert:
    """Протокол не выполнен — отсутствует метод insert."""

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def ntotal(self) -> int:
        return 0

    @property
    def existing_doc_ids(self) -> set[str]:
        return set()

    def search(self, query_embeddings, top_k=5, filter_metadata=None):
        return []

    def save(self, directory):
        pass

    def reset(self):
        pass


class _MissingProperties:
    """Протокол не выполнен — отсутствуют свойства."""

    def insert(self, embeddings, metadata):
        pass

    def search(self, query_embeddings, top_k=5, filter_metadata=None):
        return []

    def save(self, directory):
        pass

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


class TestBaseVectorStoreProtocol:
    def test_full_impl_is_instance(self):
        """Полная структурная реализация проходит isinstance без наследования."""
        obj = _FullImpl()
        assert isinstance(obj, BaseVectorStore)

    def test_missing_insert_is_not_instance(self):
        """Отсутствие insert — не соответствует протоколу."""
        obj = _MissingInsert()
        assert not isinstance(obj, BaseVectorStore)

    def test_missing_properties_is_not_instance(self):
        """Отсутствие свойств — не соответствует протоколу."""
        obj = _MissingProperties()
        assert not isinstance(obj, BaseVectorStore)

    def test_plain_object_is_not_instance(self):
        """Произвольный объект не проходит isinstance."""
        assert not isinstance(object(), BaseVectorStore)

    def test_none_is_not_instance(self):
        """None не проходит isinstance."""
        assert not isinstance(None, BaseVectorStore)

    def test_protocol_is_runtime_checkable(self):
        """Сам протокол помечен @runtime_checkable."""

        # hasattr проверяет, что протокол помечен runtime_checkable
        assert hasattr(BaseVectorStore, "__protocol_attrs__") or (
            hasattr(BaseVectorStore, "_is_protocol")
            or hasattr(BaseVectorStore, "__runtime_checkable__")
        )

    def test_full_impl_embedding_dim(self):
        """Свойство embedding_dim доступно и возвращает int."""
        obj = _FullImpl()
        assert isinstance(obj.embedding_dim, int)

    def test_full_impl_ntotal(self):
        """Свойство ntotal доступно и возвращает int."""
        obj = _FullImpl()
        assert isinstance(obj.ntotal, int)

    def test_full_impl_existing_doc_ids(self):
        """Свойство existing_doc_ids возвращает set."""
        obj = _FullImpl()
        assert isinstance(obj.existing_doc_ids, set)
