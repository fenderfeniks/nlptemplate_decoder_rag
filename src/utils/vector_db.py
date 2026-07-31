# src/utils/vector_db.py
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm


try:
    from datasketch import MinHashLSH
except ImportError:
    MinHashLSH = None
    logging.getLogger(__name__).warning(
        "Библиотека datasketch не установлена. Fuzzy-дедупликация (MinHashLSH) отключена."
    )

logger = logging.getLogger(__name__)


class FAISSVectorDB:
    """Локальная векторная база данных с поддержкой HNSW и пост-фильтрации.

    Поддерживает:
    - Два типа индекса: 'flat' (точный поиск) и 'hnsw' (приближённый).
    - Автоматическую нормализацию векторов для корректного косинусного сходства.
    - Батчевую индексацию с прогресс-баром.
    - Итеративную пост-фильтрацию по метаданным с гарантированным top_k.
    - Персистентность индекса, метаданных и состояния LSH на диске.

    Предупреждение по безопасности:
        Метод `load()` использует `pickle` для десериализации метаданных и LSH.
        Загружайте только файлы из доверенных источников — pickle может
        выполнить произвольный код при десериализации.
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
        lsh_threshold: float = 0.85,
        lsh_num_perm: int = 128,
    ) -> None:
        """
        Args:
            embedding_dim: Размерность векторов.
            index_type: Тип индекса — 'flat' или 'hnsw'.
            m: (HNSW) Число связей на узел.
            ef_construction: (HNSW) Ширина поиска при построении.
            ef_search: (HNSW) Ширина поиска при запросе.
            normalize_embeddings: Нормализовать ли векторы перед вставкой/поиском.
            insert_batch_size: Размер батча при батчевой индексации.
            filter_fetch_multiplier: Начальный множитель over-fetch при фильтрации.
            filter_max_fetch_multiplier: Максимальный множитель при итеративном расширении.
            lsh_threshold: (LSH) Порог сходства Jaccard для нечеткой дедупликации.
            lsh_num_perm: (LSH) Количество перестановок хэш-функции.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self.normalize_embeddings = normalize_embeddings
        self.insert_batch_size = insert_batch_size
        self.filter_fetch_multiplier = filter_fetch_multiplier
        self.filter_max_fetch_multiplier = filter_max_fetch_multiplier
        self.lsh_threshold = lsh_threshold
        self.lsh_num_perm = lsh_num_perm

        if self.index_type == "hnsw":
            if not self.normalize_embeddings:
                logger.warning(
                    "index_type='hnsw' использует METRIC_INNER_PRODUCT, "
                    "который равнозначен косинусному сходству ТОЛЬКО для нормализованных векторов. "
                    "Настоятельно рекомендуется normalize_embeddings=True."
                )
            logger.info(
                "Инициализация FAISS: IndexHNSWFlat (M=%d, ef_c=%d, ef_s=%d)",
                m,
                ef_construction,
                ef_search,
            )
            self.index = faiss.IndexHNSWFlat(embedding_dim, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search

        elif self.index_type == "flat":
            logger.info("Инициализация FAISS: IndexFlatIP (точный поиск)")
            self.index = faiss.IndexFlatIP(embedding_dim)

        else:
            raise ValueError(
                f"Неизвестный тип индекса: '{self.index_type}'. Поддерживаются: 'flat', 'hnsw'."
            )

        self.metadata: list[dict[str, Any]] = []

        # Кэш doc_id для O(1) проверки дублей (инвалидируется при insert/reset)
        self._doc_id_cache: set[str] | None = None

        if MinHashLSH is not None:
            self.lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.lsh_num_perm)
            logger.info("MinHashLSH инициализирован (threshold=%.2f)", self.lsh_threshold)
        else:
            self.lsh = None

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @property
    def existing_doc_ids(self) -> set[str]:
        """Множество всех doc_id, уже присутствующих в БД.

        Результат кэшируется и инвалидируется при каждом insert/reset.
        """
        if self._doc_id_cache is None:
            self._doc_id_cache = {meta["doc_id"] for meta in self.metadata if "doc_id" in meta}
        return self._doc_id_cache

    def _invalidate_cache(self) -> None:
        self._doc_id_cache = None

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-нормализует векторы по строкам. Клипует нормы во избежание деления на 0."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-10, None)

    def _prepare(self, embeddings: np.ndarray) -> np.ndarray:
        """Приводит к float32 и опционально нормализует."""
        embeddings = embeddings.astype(np.float32)
        if self.normalize_embeddings:
            embeddings = self._normalize(embeddings)
        return embeddings

    def _check_consistency(self) -> None:
        """Проверяет синхронность индекса и списка метаданных."""
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"Нарушена консистентность: index.ntotal={self.index.ntotal}, "
                f"len(metadata)={len(self.metadata)}. "
                "Возможно, insert упал на середине — вызовите reset() или load() из чекпоинта."
            )

    def _match_filters(self, doc_meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Проверяет, соответствует ли документ заданным фильтрам."""
        return all(doc_meta.get(key) == value for key, value in filters.items())

    # ------------------------------------------------------------------
    # Основные публичные методы
    # ------------------------------------------------------------------

    def insert(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Добавляет векторы в индекс.

        Args:
            embeddings: np.ndarray формы (N, embedding_dim).
            metadata: Список словарей длиной N.

        Raises:
            ValueError: При несовпадении размерностей или длин.
            RuntimeError: Если состояние индекса уже нарушено до вставки.

        Note:
            Атомарность гарантируется для пары (index, metadata): при ошибке
            index.add — metadata откатываются. LSH-состояние обновляется
            снаружи (в KnowledgeBaseIndexer) и не откатывается автоматически.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Ожидается embeddings.shape=(N, {self.embedding_dim}), "
                f"получено: {embeddings.shape}"
            )
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Несоответствие длин: embeddings={len(embeddings)}, metadata={len(metadata)}"
            )

        self._check_consistency()

        prepared = self._prepare(embeddings)

        snapshot_len = len(self.metadata)
        self.metadata.extend(metadata)
        try:
            self.index.add(prepared)
        except Exception:
            del self.metadata[snapshot_len:]
            logger.exception("index.add упал — metadata откачены, индекс не изменён.")
            raise

        # Инвалидируем кэш после успешной вставки
        self._invalidate_cache()

    def insert_batched(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        desc: str = "Indexing",
    ) -> None:
        """Батчевая вставка с прогресс-баром.

        Args:
            embeddings: np.ndarray формы (N, embedding_dim).
            metadata: Список словарей длиной N.
            desc: Описание в tqdm.
        """
        n = len(embeddings)
        for start in tqdm(range(0, n, self.insert_batch_size), desc=desc, unit="batch"):
            end = min(start + self.insert_batch_size, n)
            self.insert(embeddings[start:end], metadata[start:end])

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

        prepared_queries = self._prepare(query_embeddings)

        if not filter_metadata:
            return self._search_no_filter(prepared_queries, top_k)

        return self._search_with_filter(prepared_queries, top_k, filter_metadata)

    # ------------------------------------------------------------------
    # Внутренние методы поиска
    # ------------------------------------------------------------------

    def _search_no_filter(
        self,
        prepared_queries: np.ndarray,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        fetch_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(prepared_queries, fetch_k)

        return [
            [
                {"score": float(d), "metadata": self.metadata[i]}
                for d, i in zip(dist_row, idx_row)
                if i != -1
            ]
            for dist_row, idx_row in zip(distances, indices)
        ]

    def _search_with_filter(
        self,
        prepared_queries: np.ndarray,
        top_k: int,
        filter_metadata: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        """Итеративный over-fetch: удваивает fetch_k пока не наберём top_k или не упрёмся в ntotal.

        Оптимизация: при каждой итерации ищем только незавершённые запросы,
        чтобы не тратить время на уже набравшие top_k.
        """
        multiplier = self.filter_fetch_multiplier
        max_multiplier = self.filter_max_fetch_multiplier

        n_queries = len(prepared_queries)
        batch_results: list[list[dict[str, Any]]] = [[] for _ in range(n_queries)]

        # Маска незавершённых запросов: индекс в оригинальном массиве → индекс в активном батче
        pending_original_indices = list(range(n_queries))

        while pending_original_indices and multiplier <= max_multiplier:
            fetch_k = min(top_k * multiplier, self.index.ntotal)

            active_queries = prepared_queries[pending_original_indices]
            distances, indices = self.index.search(active_queries, fetch_k)

            still_pending = []
            for local_idx, orig_idx in enumerate(pending_original_indices):
                dist_row = distances[local_idx]
                idx_row = indices[local_idx]

                row_res: list[dict[str, Any]] = []
                for d, i in zip(dist_row, idx_row):
                    if i == -1:
                        continue
                    if self._match_filters(self.metadata[i], filter_metadata):
                        row_res.append({"score": float(d), "metadata": self.metadata[i]})
                    if len(row_res) == top_k:
                        break

                batch_results[orig_idx] = row_res

                if len(row_res) < top_k and fetch_k < self.index.ntotal:
                    still_pending.append(orig_idx)

            pending_original_indices = still_pending

            if pending_original_indices:
                multiplier *= 2
                logger.debug(
                    "Фильтрация: %d запросов не набрали top_k=%d, расширяем до multiplier=%d",
                    len(pending_original_indices),
                    top_k,
                    multiplier,
                )

        return batch_results

    # ------------------------------------------------------------------
    # Персистентность
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Сохраняет индекс, метаданные и состояние LSH на диск."""
        self._check_consistency()
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(dir_path / "index.faiss"))

        with open(dir_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.lsh is not None:
            with open(dir_path / "lsh.pkl", "wb") as f:
                pickle.dump(self.lsh, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("VectorDB сохранена в '%s' (%d векторов).", directory, self.index.ntotal)

    @classmethod
    def load(cls, directory: str | Path, **init_kwargs: Any) -> "FAISSVectorDB":
        """Загружает индекс, метаданные и состояние LSH с диска.

        Warning:
            Использует pickle. Загружайте только файлы из доверенных источников.
        """
        dir_path = Path(directory)
        index_path = dir_path / "index.faiss"
        meta_path = dir_path / "metadata.pkl"
        lsh_path = dir_path / "lsh.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Файл индекса не найден: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {meta_path}")

        warnings.warn(
            "FAISSVectorDB.load() использует pickle для десериализации. "
            "Убедитесь, что файлы получены из доверенного источника.",
            UserWarning,
            stacklevel=2,
        )

        instance = cls(**init_kwargs)
        instance.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            instance.metadata = pickle.load(f)  # noqa: S301

        if instance.lsh is not None and lsh_path.exists():
            with open(lsh_path, "rb") as f:
                instance.lsh = pickle.load(f)  # noqa: S301

        instance._check_consistency()
        logger.info("VectorDB загружена из '%s' (%d векторов).", directory, instance.index.ntotal)
        return instance

    def reset(self) -> None:
        """Полностью очищает индекс, метаданные и LSH."""
        self.index.reset()
        self.metadata = []
        self._invalidate_cache()
        if self.lsh is not None:
            self.lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.lsh_num_perm)
        logger.info("VectorDB сброшена.")
