# src/vector_store/qdrant_store.py
"""Qdrant-бэкенд векторного хранилища.

Реализует ``BaseVectorStore``. Для смены с FAISS на Qdrant достаточно
поменять ``- vector_db: flat`` на ``- vector_db: qdrant`` в main.yaml —
остальной код не трогается.

Требует: ``pip install qdrant-client``

Отличия от FAISS в части протокола:
- ``save()`` — no-op: данные уже персистентны на сервере Qdrant.
- ``reset()`` — удаляет и пересоздаёт коллекцию (аналог index.reset()).
- ``load()`` / ``connect()`` — подключение к существующей коллекции,
  ``directory`` игнорируется (аргумент принимается для унификации с FAISS
  через ``hydra.utils.instantiate(cfg.vector_db.loader, directory=...)``.
- Фильтрация по метаданным — нативная через Qdrant Filter (эффективнее
  чем post-filtering в FAISS).
- ``existing_doc_ids`` — scroll по всей коллекции; кэшируется, инвалидируется
  при upsert/reset.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore[assignment, misc]
    qmodels = None  # type: ignore[assignment, misc]


def _require_qdrant() -> None:
    if not _QDRANT_AVAILABLE:
        raise ImportError("qdrant-client не установлен. Установите: pip install qdrant-client")


class QdrantVectorStore:
    """Векторное хранилище на базе Qdrant.

    Реализует протокол ``BaseVectorStore``.

    Поддерживает:
    - Cosine / Dot / Euclidean метрики через параметр ``distance``.
    - Нативную фильтрацию по метаданным через Qdrant Filter — без over-fetch.
    - Опциональный in-memory режим (``url=":memory:"`` или ``in_memory=True``)
      для тестов без запущенного сервера.
    - Батчевую вставку с контролем размера батча.
    - ``existing_doc_ids`` с кэшированием для инкрементальной индексации.
    """

    def __init__(
        self,
        embedding_dim: int,
        collection_name: str = "knowledge_base",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        distance: str = "Cosine",
        normalize_embeddings: bool = True,
        insert_batch_size: int = 256,
        in_memory: bool = False,
        recreate_collection: bool = False,
    ) -> None:
        """
        Args:
            embedding_dim: Размерность векторов. Должна совпадать
                с размерностью энкодера.
            collection_name: Имя коллекции в Qdrant.
            url: URL Qdrant-сервера. ``':memory:'`` или ``in_memory=True``
                для локального in-memory режима (тесты, dev).
            api_key: API-ключ для Qdrant Cloud (опционально).
            distance: Метрика расстояния — ``'Cosine'``, ``'Dot'``, ``'Euclid'``.
                ``'Cosine'`` автоматически нормализует векторы на стороне Qdrant,
                но мы нормализуем на своей стороне тоже если ``normalize_embeddings=True``
                — двойная нормализация идемпотентна для единичных векторов.
            normalize_embeddings: Нормализовать ли векторы перед вставкой/поиском
                на клиентской стороне. Рекомендуется ``True`` для ``Cosine`` и ``Dot``.
            insert_batch_size: Размер батча для ``upsert`` (Qdrant рекомендует 100-256).
            in_memory: Использовать локальный in-memory клиент (без сервера).
                Полезно для unit-тестов и быстрого прототипирования.
            recreate_collection: Удалить и пересоздать коллекцию при инициализации
                если она уже существует. Используйте осторожно в проде.

        Raises:
            ImportError: Если qdrant-client не установлен.
            ValueError: При неизвестном значении ``distance``.
        """
        _require_qdrant()

        _valid_distances = {"Cosine", "Dot", "Euclid"}
        if distance not in _valid_distances:
            raise ValueError(
                f"Неизвестная метрика: '{distance}'. Допустимые: {sorted(_valid_distances)}."
            )

        self._embedding_dim = embedding_dim
        self.collection_name = collection_name
        self.distance = distance
        self.normalize_embeddings = normalize_embeddings
        self.insert_batch_size = insert_batch_size

        # Инициализация клиента
        if in_memory or url == ":memory:":
            logger.info("QdrantVectorStore: in-memory режим (без сервера).")
            self._client = QdrantClient(":memory:")
        else:
            logger.info(
                "QdrantVectorStore: подключение к %s (collection='%s')", url, collection_name
            )
            self._client = QdrantClient(url=url, api_key=api_key, timeout=30)

        self._doc_id_cache: set[str] | None = None
        self._ensure_collection(recreate=recreate_collection)

    # ------------------------------------------------------------------
    # Инициализация коллекции
    # ------------------------------------------------------------------

    def _ensure_collection(self, recreate: bool = False) -> None:
        """Создаёт коллекцию если она не существует. При recreate=True — пересоздаёт."""
        distance_map = {
            "Cosine": qmodels.Distance.COSINE,
            "Dot": qmodels.Distance.DOT,
            "Euclid": qmodels.Distance.EUCLID,
        }
        qdrant_distance = distance_map[self.distance]

        collections = {c.name for c in self._client.get_collections().collections}

        if recreate and self.collection_name in collections:
            logger.warning(
                "QdrantVectorStore: удаляем коллекцию '%s' (recreate=True).",
                self.collection_name,
            )
            self._client.delete_collection(self.collection_name)
            collections.discard(self.collection_name)

        if self.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._embedding_dim,
                    distance=qdrant_distance,
                ),
            )
            logger.info(
                "Коллекция '%s' создана (dim=%d, distance=%s).",
                self.collection_name,
                self._embedding_dim,
                self.distance,
            )
        else:
            logger.info("Коллекция '%s' уже существует — подключаемся.", self.collection_name)

    # ------------------------------------------------------------------
    # Свойства состояния (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def ntotal(self) -> int:
        """Общее число точек в коллекции."""
        return self._client.count(collection_name=self.collection_name).count

    @property
    def existing_doc_ids(self) -> set[str]:
        """Все ``doc_id`` в коллекции. Кэшируется, инвалидируется при upsert/reset."""
        if self._doc_id_cache is not None:
            return self._doc_id_cache

        doc_ids: set[str] = set()
        next_offset = None

        # Scroll по всей коллекции батчами — коллекция может быть большой
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=1000,
                offset=next_offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for record in records:
                if record.payload and "doc_id" in record.payload:
                    doc_ids.add(record.payload["doc_id"])

            if next_offset is None:
                break

        self._doc_id_cache = doc_ids
        return self._doc_id_cache

    def _invalidate_cache(self) -> None:
        self._doc_id_cache = None

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-10, None)

    def _prepare(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = embeddings.astype(np.float32)
        if self.normalize_embeddings:
            embeddings = self._normalize(embeddings)
        return embeddings

    def _build_filter(self, filter_metadata: dict[str, Any] | None) -> qmodels.Filter | None:
        """Конвертирует dict фильтров в нативный Qdrant Filter (AND-семантика)."""
        if not filter_metadata:
            return None

        conditions = [
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=value),
            )
            for key, value in filter_metadata.items()
        ]
        return qmodels.Filter(must=conditions)

    # ------------------------------------------------------------------
    # Запись (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    def insert(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Вставляет векторы в Qdrant батчами через upsert.

        Raises:
            ValueError: При несоответствии размерностей или длин.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Ожидается embeddings.shape=(N, {self._embedding_dim}), "
                f"получено: {embeddings.shape}"
            )
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Несоответствие длин: embeddings={len(embeddings)}, metadata={len(metadata)}"
            )

        prepared = self._prepare(embeddings)

        for start in range(0, len(prepared), self.insert_batch_size):
            end = min(start + self.insert_batch_size, len(prepared))
            batch_emb = prepared[start:end]
            batch_meta = metadata[start:end]

            points = [
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload=meta,
                )
                for emb, meta in zip(batch_emb, batch_meta)  # noqa
            ]
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        self._invalidate_cache()
        logger.debug("Upsert завершён: %d точек в '%s'.", len(embeddings), self.collection_name)

    # ------------------------------------------------------------------
    # Поиск (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Поиск ближайших векторов с опциональной нативной фильтрацией."""
        if self.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        prepared = self._prepare(query_embeddings)
        qdrant_filter = self._build_filter(filter_metadata)

        results = []
        for query_vec in prepared:
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vec.tolist(),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            results.append(
                [
                    {
                        "score": float(hit.score),
                        "metadata": hit.payload or {},
                    }
                    for hit in hits
                ]
            )

        return results

    # ------------------------------------------------------------------
    # Персистентность (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    def save(self, directory: str | Path | None = None) -> None:
        """No-op: данные уже персистентны на сервере Qdrant."""
        logger.info(
            "QdrantVectorStore.save() вызван — данные уже персистентны в Qdrant "
            "(коллекция '%s'). Никаких дополнительных действий не требуется.",
            self.collection_name,
        )

    def reset(self) -> None:
        """Удаляет и пересоздаёт коллекцию — полная очистка."""
        self._client.delete_collection(self.collection_name)
        self._ensure_collection(recreate=False)
        self._invalidate_cache()
        logger.info("QdrantVectorStore: коллекция '%s' сброшена.", self.collection_name)

    # ------------------------------------------------------------------
    # Фабричный метод для загрузки через hydra.utils.instantiate
    # ------------------------------------------------------------------

    @classmethod
    def connect(
        cls,
        directory: str | Path | None = None,
        **kwargs: Any,
    ) -> QdrantVectorStore:
        """Подключается к существующей коллекции Qdrant.

        Используется через ``cfg.vector_db.loader`` при старте сервера —
        аналог ``FAISSVectorStore.load()``.

        Фильтрует ``**kwargs`` через сигнатуру ``__init__`` — лишние ключи
        которые Hydra подмешивает из конфига (например из storage-интерполяций)
        отсекаются с предупреждением, не вызывая TypeError.

        Args:
            directory: Игнорируется. Принимается для совместимости с FAISS-интерфейсом
                при вызове ``hydra.utils.instantiate(cfg.vector_db.loader, directory=...)``.
            **kwargs: Параметры для ``__init__``. Неизвестные ключи игнорируются.

        Returns:
            Инициализированный ``QdrantVectorStore`` подключённый к коллекции.
        """
        if directory is not None:
            logger.debug(
                "QdrantVectorStore.connect(): аргумент directory='%s' игнорируется "
                "(Qdrant не использует локальные файлы индекса).",
                directory,
            )

        # Фильтруем kwargs через сигнатуру __init__ — та же защита что в FAISSVectorStore.load.
        # Hydra может подмешать лишние ключи из storage-интерполяций конфига.
        valid_keys = set(inspect.signature(cls.__init__).parameters) - {"self"}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        dropped = set(kwargs) - valid_keys
        if dropped:
            logger.warning(
                "QdrantVectorStore.connect: игнорируем неизвестные kwargs из конфига: %s",
                dropped,
            )

        instance = cls(**filtered, recreate_collection=False)

        ntotal = instance.ntotal
        if ntotal == 0:
            logger.warning(
                "QdrantVectorStore.connect(): коллекция '%s' пуста. "
                "Запустите индексацию перед стартом сервера.",
                instance.collection_name,
            )
        else:
            logger.info(
                "QdrantVectorStore подключён к '%s' (%d документов).",
                instance.collection_name,
                ntotal,
            )

        return instance
