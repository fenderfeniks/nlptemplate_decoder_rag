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
- ``get_uri()`` — возвращает ``qdrant://<url>/<collection>`` для записи в манифест.
  ``ArtifactResolver`` распознаёт эту схему и не пытается скачивать файлы.
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
    - ``get_uri()`` для записи адреса коллекции в манифест вместо пути к файлам.
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
        enable_sparse: bool = False,
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
        self.enable_sparse = enable_sparse

        self._url = url
        self._in_memory = in_memory or url == ":memory:"

        if self._in_memory:
            logger.info("QdrantVectorStore: in-memory режим (без сервера).")
            self._client = QdrantClient(":memory:")
        else:
            logger.info(
                "QdrantVectorStore: подключение к %s (collection='%s')", url, collection_name
            )
            self._client = QdrantClient(url=url, api_key=api_key, timeout=30)

        # Sparse-модель инициализируется один раз и живёт в store
        self._sparse_model = None
        if enable_sparse:
            try:
                from fastembed import SparseTextEmbedding
                self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                logger.info("QdrantVectorStore: sparse-модель (BM25) загружена.")
            except ImportError:
                logger.warning(
                    "fastembed не установлен — sparse отключён. "
                    "Установите: pip install fastembed"
                )
                self.enable_sparse = False

        self._doc_id_cache: set[str] | None = None
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Инициализация коллекции
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Создаёт коллекцию если она не существует."""
        distance_map = {
            "Cosine": qmodels.Distance.COSINE,
            "Dot": qmodels.Distance.DOT,
            "Euclid": qmodels.Distance.EUCLID,
        }
        qdrant_distance = distance_map[self.distance]
        collections = {c.name for c in self._client.get_collections().collections}

        if self.collection_name not in collections:
            if self.enable_sparse:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": qmodels.VectorParams(
                            size=self._embedding_dim,
                            distance=qdrant_distance,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": qmodels.SparseVectorParams(
                            index=qmodels.SparseIndexParams(on_disk=False)
                        )
                    },
                )
                logger.info(
                    "Коллекция '%s' создана (dim=%d, distance=%s, sparse=BM25).",
                    self.collection_name, self._embedding_dim, self.distance,
                )
            else:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self._embedding_dim,
                        distance=qdrant_distance,
                    ),
                )
                logger.info(
                    "Коллекция '%s' создана (dim=%d, distance=%s, sparse=off).",
                    self.collection_name, self._embedding_dim, self.distance,
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
    # URI для манифеста
    # ------------------------------------------------------------------

    def get_uri(self) -> str:
        """Возвращает URI коллекции для записи в манифест.

        Формат: ``qdrant://<server_url>/<collection_name>``
        Пример: ``qdrant://http://localhost:6333/nlp_project_kb``

        ``ArtifactResolver`` распознаёт схему ``qdrant://`` и вместо скачивания
        файлов передаёт url и collection_name напрямую в ``QdrantVectorStore.connect()``.

        In-memory режим возвращает ``qdrant+memory:///<collection_name>`` —
        используется только в тестах, в реальном манифесте не появляется.
        """
        if self._in_memory:
            return f"qdrant+memory:///{self.collection_name}"
        return f"qdrant://{self._url}/{self.collection_name}"

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

        Если enable_sparse=True — вычисляет BM25 sparse-векторы из metadata['text']
        и сохраняет оба вектора в именованных полях 'dense' и 'sparse'.

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

        # Вычисляем sparse-векторы один раз для всего батча
        sparse_vectors = None
        if self.enable_sparse and self._sparse_model is not None:
            texts = [m.get("text", "") for m in metadata]
            sparse_vectors = list(self._sparse_model.embed(texts))

        for start in range(0, len(prepared), self.insert_batch_size):
            end = min(start + self.insert_batch_size, len(prepared))
            batch_emb = prepared[start:end]
            batch_meta = metadata[start:end]

            if sparse_vectors is not None:
                batch_sparse = sparse_vectors[start:end]
                points = [
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": emb.tolist(),
                            "sparse": qmodels.SparseVector(
                                indices=sv.indices.tolist(),
                                values=sv.values.tolist(),
                            ),
                        },
                        payload=meta,
                    )
                    for emb, sv, meta in zip(batch_emb, batch_sparse, batch_meta)
                ]
            else:
                points = [
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload=meta,
                    )
                    for emb, meta in zip(batch_emb, batch_meta)
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
        """Поиск ближайших векторов с опциональной нативной фильтрацией.

        Если enable_sparse=True — использует именованный вектор 'dense'.
        Если enable_sparse=False — безымянный вектор (обратная совместимость).
        """
        if self.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        prepared = self._prepare(query_embeddings)
        qdrant_filter = self._build_filter(filter_metadata)

        results = []
        for query_vec in prepared:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec.tolist(),
                using="dense" if self.enable_sparse else None,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            results.append(
                [
                    {
                        "score": float(hit.score),
                        "metadata": hit.payload or {},
                    }
                    for hit in response.points
                ]
            )

        return results

    def search_hybrid(
        self,
        query_vectors: np.ndarray,
        query_texts: list[str],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Гибридный поиск (Dense + Sparse) с Reciprocal Rank Fusion (RRF).

        Требует enable_sparse=True и коллекции с именованными векторами.

        Raises:
            RuntimeError: Если вызван при enable_sparse=False.
        """
        if not self.enable_sparse or self._sparse_model is None:
            raise RuntimeError(
                "search_hybrid недоступен: enable_sparse=False. "
                "Используйте search() или включите sparse в конфиге."
            )

        if self.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]

        prepared_dense = self._prepare(query_vectors)
        qdrant_filter = self._build_filter(filter_metadata)

        # Sparse-модель уже живёт в self._sparse_model — не создаём заново
        sparse_vectors = list(self._sparse_model.embed(query_texts))

        final_results = []
        prefetch_limit = top_k * 2

        for dense_vec, sparse_vec in zip(prepared_dense, sparse_vectors):
            qdrant_sparse = qmodels.SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist(),
            )

            response = self._client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    qmodels.Prefetch(
                        query=dense_vec.tolist(),
                        using="dense",
                        limit=prefetch_limit,
                        filter=qdrant_filter,
                    ),
                    qmodels.Prefetch(
                        query=qdrant_sparse,
                        using="sparse",
                        limit=prefetch_limit,
                        filter=qdrant_filter,
                    ),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            final_results.append(
                [
                    {
                        "score": float(hit.score),
                        "metadata": hit.payload or {},
                        "doc_id": hit.payload.get("doc_id") if hit.payload else str(hit.id),
                    }
                    for hit in response.points
                ]
            )

        return final_results

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
        self._ensure_collection()  # убрали recreate=False — метод его не принимает
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

        Используется через ``cfg.vector_db.loader`` при старте сервера.
        Умеет парсить URI формата ``qdrant://<url>/<collection>`` который
        ``ArtifactResolver`` передаёт через аргумент ``directory``.

        Args:
            directory: Либо локальный путь (игнорируется), либо URI вида
                ``qdrant://http://localhost:6333/nlp_project_kb`` — тогда
                из него извлекаются ``url`` и ``collection_name``.
            **kwargs: Параметры для ``__init__``. Неизвестные ключи игнорируются.
        """
        if directory is not None:
            uri = str(directory)
            if uri.startswith("qdrant://"):
                # Парсим: qdrant://http://localhost:6333/nlp_project_kb
                #      -> url=http://localhost:6333, collection=nlp_project_kb
                without_scheme = uri[len("qdrant://"):]
                last_slash = without_scheme.rfind("/")
                if last_slash == -1:
                    raise ValueError(
                        f"Невалидный qdrant:// URI — не найден '/' после хоста: {uri}"
                    )
                parsed_url = without_scheme[:last_slash]
                parsed_collection = without_scheme[last_slash + 1:]
                kwargs.setdefault("url", parsed_url)
                kwargs.setdefault("collection_name", parsed_collection)
                logger.info(
                    "QdrantVectorStore.connect(): из URI — url='%s', collection='%s'",
                    parsed_url,
                    parsed_collection,
                )
            else:
                logger.debug(
                    "QdrantVectorStore.connect(): directory='%s' не является qdrant:// URI — "
                    "игнорируется (Qdrant не использует локальные файлы индекса).",
                    directory,
                )

        valid_keys = set(inspect.signature(cls.__init__).parameters) - {"self"}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        dropped = set(kwargs) - valid_keys
        if dropped:
            logger.warning(
                "QdrantVectorStore.connect: игнорируем неизвестные kwargs из конфига: %s",
                dropped,
            )

        instance = cls(**filtered)

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