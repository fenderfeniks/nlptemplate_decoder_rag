# src/vector_store/faiss_store.py
"""FAISS-бэкенд векторного хранилища.

Реализует ``BaseVectorStore``. Для смены бэкенда на Qdrant — замените
этот класс на ``QdrantVectorStore`` в конфиге, интерфейс не изменится.

Метаданные сохраняются в JSON (не pickle) — безопасный формат без риска
выполнения произвольного кода при десериализации.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

_VALID_INDEX_TYPES = frozenset({"flat", "hnsw"})


class FAISSVectorStore:
    """Локальное векторное хранилище на базе FAISS.

    Реализует протокол ``BaseVectorStore``.

    Поддерживает:
    - Два типа индекса: ``'flat'`` (точный, для eval) и ``'hnsw'`` (приближённый, для prod).
    - Опциональную L2-нормализацию векторов перед вставкой/поиском.
    - Батчевую индексацию с прогресс-баром.
    - Итеративную пост-фильтрацию по метаданным с гарантированным top_k.
    - Персистентность индекса (FAISS-бинарник) и метаданных (JSON).

    LSH-дедупликация вынесена в ``src.vector_store.lsh.LSHIndex`` —
    она не является ответственностью хранилища.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        normalize_embeddings: bool = True,
        insert_batch_size: int = 10_000,
        filter_fetch_multiplier: int = 5,
        filter_max_fetch_multiplier: int = 50,
    ) -> None:
        """
        Args:
            embedding_dim: Размерность векторов.
            index_type: Тип индекса — ``'flat'`` или ``'hnsw'``.
            m: (HNSW) Число связей на узел. Больше -> точнее, больше RAM.
            ef_construction: (HNSW) Ширина поиска при построении.
            ef_search: (HNSW) Ширина поиска при запросе.
            normalize_embeddings: Нормализовать ли векторы перед вставкой/поиском.
                Обязательно ``True`` для корректного косинусного сходства через
                ``IndexFlatIP`` / ``IndexHNSWFlat`` с ``METRIC_INNER_PRODUCT``.
            insert_batch_size: Размер батча для ``insert_batched``.
            filter_fetch_multiplier: Начальный over-fetch множитель при фильтрации.
            filter_max_fetch_multiplier: Максимальный множитель при итеративном расширении.

        Raises:
            ValueError: При неизвестном ``index_type``.
        """
        if index_type.lower() not in _VALID_INDEX_TYPES:
            raise ValueError(
                f"Неизвестный тип индекса: '{index_type}'. "
                f"Поддерживаются: {sorted(_VALID_INDEX_TYPES)}."
            )

        self._embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self.normalize_embeddings = normalize_embeddings
        self.insert_batch_size = insert_batch_size
        self.filter_fetch_multiplier = filter_fetch_multiplier
        self.filter_max_fetch_multiplier = filter_max_fetch_multiplier

        self.index = self._build_index(embedding_dim, index_type, m, ef_construction, ef_search)
        self._metadata: list[dict[str, Any]] = []
        self._doc_id_cache: set[str] | None = None

    @staticmethod
    def _build_index(
        embedding_dim: int,
        index_type: str,
        m: int,
        ef_construction: int,
        ef_search: int,
    ) -> faiss.Index:
        index_type = index_type.lower()
        if index_type == "hnsw":
            logger.info(
                "FAISS: IndexHNSWFlat (dim=%d, M=%d, ef_c=%d, ef_s=%d)",
                embedding_dim,
                m,
                ef_construction,
                ef_search,
            )
            idx = faiss.IndexHNSWFlat(embedding_dim, m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = ef_construction
            idx.hnsw.efSearch = ef_search
            return idx

        logger.info("FAISS: IndexFlatIP (dim=%d, точный поиск)", embedding_dim)
        return faiss.IndexFlatIP(embedding_dim)

    # ------------------------------------------------------------------
    # Свойства состояния (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def existing_doc_ids(self) -> set[str]:
        """Кэшированное множество ``doc_id``. Инвалидируется при insert/reset."""
        if self._doc_id_cache is None:
            self._doc_id_cache = {m["doc_id"] for m in self._metadata if "doc_id" in m}
        return self._doc_id_cache

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._doc_id_cache = None

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-10, None)

    def _prepare(self, embeddings: np.ndarray) -> np.ndarray:
        """Приводит к float32 и опционально нормализует."""
        embeddings = embeddings.astype(np.float32)
        if self.normalize_embeddings:
            embeddings = self._normalize(embeddings)
        return embeddings

    def _check_consistency(self) -> None:
        if self.index.ntotal != len(self._metadata):
            raise RuntimeError(
                f"Нарушена консистентность: index.ntotal={self.index.ntotal}, "
                f"len(metadata)={len(self._metadata)}. "
                "Вызовите reset() или load() из чекпоинта."
            )

    def _match_filters(self, doc_meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        return all(doc_meta.get(k) == v for k, v in filters.items())

    # ------------------------------------------------------------------
    # Запись (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    def insert(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Добавляет векторы в индекс.

        Атомарность для пары (index, metadata): при ошибке ``index.add``
        метаданные откатываются к состоянию до вызова.

        Raises:
            ValueError: При несовпадении размерностей или длин.
            RuntimeError: Если состояние индекса уже нарушено до вставки.
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

        self._check_consistency()
        prepared = self._prepare(embeddings)

        snapshot_len = len(self._metadata)
        self._metadata.extend(metadata)
        try:
            self.index.add(prepared)
        except Exception:
            del self._metadata[snapshot_len:]
            logger.exception("index.add упал — metadata откачены, индекс не изменён.")
            raise

        self._invalidate_cache()

    def insert_batched(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        desc: str = "Indexing",
    ) -> None:
        """Батчевая вставка с прогресс-баром."""
        n = len(embeddings)
        for start in tqdm(range(0, n, self.insert_batch_size), desc=desc, unit="batch"):
            end = min(start + self.insert_batch_size, n)
            self.insert(embeddings[start:end], metadata[start:end])

    # ------------------------------------------------------------------
    # Поиск (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Поиск ближайших векторов с опциональной пост-фильтрацией."""
        self._check_consistency()

        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        prepared = self._prepare(query_embeddings)

        if not filter_metadata:
            return self._search_no_filter(prepared, top_k)
        return self._search_with_filter(prepared, top_k, filter_metadata)

    def _search_no_filter(
        self,
        prepared_queries: np.ndarray,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        fetch_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(prepared_queries, fetch_k)

        return [
            [
                {"score": float(d), "metadata": self._metadata[i]}
                for d, i in zip(dist_row, idx_row, strict=True)
                if i != -1
            ]
            for dist_row, idx_row in zip(distances, indices, strict=True)
        ]

    def _search_with_filter(
        self,
        prepared_queries: np.ndarray,
        top_k: int,
        filter_metadata: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        """Итеративный over-fetch: удваивает fetch_k пока не наберём top_k.

        Оптимизация: ищем только незавершённые запросы на каждой итерации.
        """
        multiplier = self.filter_fetch_multiplier
        n_queries = len(prepared_queries)
        batch_results: list[list[dict[str, Any]]] = [[] for _ in range(n_queries)]
        pending = list(range(n_queries))

        while pending and multiplier <= self.filter_max_fetch_multiplier:
            fetch_k = min(top_k * multiplier, self.index.ntotal)
            distances, indices = self.index.search(prepared_queries[pending], fetch_k)

            still_pending = []
            for local_idx, orig_idx in enumerate(pending):
                row_res: list[dict[str, Any]] = []
                for d, i in zip(distances[local_idx], indices[local_idx], strict=True):
                    if i == -1:
                        continue
                    if self._match_filters(self._metadata[i], filter_metadata):
                        row_res.append({"score": float(d), "metadata": self._metadata[i]})
                    if len(row_res) == top_k:
                        break

                batch_results[orig_idx] = row_res

                if len(row_res) < top_k and fetch_k < self.index.ntotal:
                    still_pending.append(orig_idx)

            pending = still_pending
            if pending:
                multiplier *= 2
                logger.debug(
                    "Фильтрация: %d запросов не набрали top_k=%d -> multiplier=%d",
                    len(pending),
                    top_k,
                    multiplier,
                )

        return batch_results

    # ------------------------------------------------------------------
    # Персистентность (BaseVectorStore protocol)
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Сохраняет индекс (FAISS-бинарник) и метаданные (JSON) на диск."""
        self._check_consistency()
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Обход бага FAISS на Windows с не-ASCII путями:
        # сериализуем индекс в памяти и пишем на диск средствами Python.
        index_bytes = faiss.serialize_index(self.index).tobytes()
        with open(dir_path / "index.faiss", "wb") as f:
            f.write(index_bytes)

        # JSON вместо pickle — безопасный формат, читается любым инструментом,
        # не выполняет произвольный код при десериализации.
        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

        logger.info(
            "FAISSVectorStore сохранён в '%s' (%d векторов).",
            directory,
            self.index.ntotal,
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        **kwargs,
    ) -> FAISSVectorStore:
        """Загружает индекс и метаданные с диска.

        Принимает ``**kwargs`` для совместимости с ``hydra.utils.instantiate`` —
        Hydra передаёт все ключи из ``cfg.vector_db.loader`` включая возможные
        лишние (например, из storage-интерполяций). Лишние ключи фильтруются
        через сигнатуру ``__init__`` и логируются как warning.

        Args:
            directory: Директория с файлами ``index.faiss`` и ``metadata.json``.
            **kwargs: Параметры для ``__init__`` (``embedding_dim`` обязателен).
                Неизвестные ключи игнорируются с предупреждением.
        """
        dir_path = Path(directory)
        index_path = dir_path / "index.faiss"
        meta_path = dir_path / "metadata.json"

        # Обратная совместимость: если json не найден, пробуем старый pkl.
        # Позволяет плавно мигрировать существующие индексы без переиндексации.
        legacy_meta_path = dir_path / "metadata.pkl"
        use_legacy = not meta_path.exists() and legacy_meta_path.exists()

        if not index_path.exists():
            raise FileNotFoundError(f"Файл индекса не найден: {index_path}")
        if not meta_path.exists() and not use_legacy:
            raise FileNotFoundError(f"Файл метаданных не найден: {meta_path}")

        # Фильтруем kwargs — оставляем только параметры __init__.
        # Hydra может подмешать лишние ключи из конфига (например 'url'
        # из storage-интерполяции в storage_router).
        valid_keys = set(inspect.signature(cls.__init__).parameters) - {"self"}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        dropped = set(kwargs) - valid_keys
        if dropped:
            logger.warning(
                "FAISSVectorStore.load: игнорируем неизвестные kwargs из конфига: %s", dropped
            )

        instance = cls(**filtered)
        with open(index_path, "rb") as f:
            index_bytes = f.read()

        instance.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))

        if use_legacy:
            import pickle  # noqa: S403

            logger.warning(
                "metadata.json не найден — загружаем legacy metadata.pkl. "
                "Запустите переиндексацию чтобы мигрировать на JSON."
            )
            with open(legacy_meta_path, "rb") as f:
                instance._metadata = pickle.load(f)  # noqa: S301
        else:
            with open(meta_path, encoding="utf-8") as f:
                instance._metadata = json.load(f)

        instance._check_consistency()
        logger.info(
            "FAISSVectorStore загружен из '%s' (%d векторов).",
            directory,
            instance.index.ntotal,
        )
        return instance

    def reset(self) -> None:
        """Полностью очищает индекс и метаданные."""
        self.index.reset()
        self._metadata = []
        self._invalidate_cache()
        logger.info("FAISSVectorStore сброшен.")
