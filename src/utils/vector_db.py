import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


class FAISSVectorDB:
    """Локальная векторная база данных с поддержкой HNSW и пост-фильтрации.

    Поддерживает:
    - Два типа индекса: 'flat' (точный поиск) и 'hnsw' (приближённый).
    - Автоматическую нормализацию векторов для корректного косинусного сходства.
    - Батчевую индексацию с прогресс-баром.
    - Итеративную пост-фильтрацию по метаданным с гарантированным top_k.
    - Персистентность индекса и метаданных на диске.
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
            index_type: Тип индекса — 'flat' или 'hnsw'.
            m: (HNSW) Число связей на узел.
            ef_construction: (HNSW) Ширина поиска при построении.
            ef_search: (HNSW) Ширина поиска при запросе.
            normalize_embeddings: Нормализовать ли векторы перед вставкой/поиском.
                Обязательно True при index_type='hnsw', т.к. METRIC_INNER_PRODUCT
                эквивалентен косинусному сходству только для единичных векторов.
            insert_batch_size: Размер батча при батчевой индексации.
            filter_fetch_multiplier: Начальный множитель over-fetch при фильтрации.
            filter_max_fetch_multiplier: Максимальный множитель при итеративном расширении.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self.normalize_embeddings = normalize_embeddings
        self.insert_batch_size = insert_batch_size
        self.filter_fetch_multiplier = filter_fetch_multiplier
        self.filter_max_fetch_multiplier = filter_max_fetch_multiplier

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

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

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
        for key, value in filters.items():
            if doc_meta.get(key) != value:
                return False
        return True

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

        # Атомарный insert: сначала расширяем metadata, потом добавляем в индекс.
        # При исключении из index.add — откатываем metadata до исходного состояния.
        snapshot_len = len(self.metadata)
        self.metadata.extend(metadata)
        try:
            self.index.add(prepared)
        except Exception:
            # Откат metadata до состояния до вставки
            del self.metadata[snapshot_len:]
            logger.exception("index.add упал — metadata откачены, индекс не изменён.")
            raise

    def insert_batched(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        desc: str = "Indexing",
    ) -> None:
        """Батчевая вставка с прогресс-баром. Удобна при индексации больших корпусов.

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
        """Поиск ближайших векторов с опциональной пост-фильтрацией.

        При наличии фильтра использует итеративный over-fetch: начинает с
        fetch_multiplier * top_k кандидатов и удваивает множитель до тех пор,
        пока не наберёт top_k подходящих или не исчерпает индекс.

        Args:
            query_embeddings: np.ndarray формы (Q, embedding_dim).
            top_k: Желаемое количество результатов на запрос.
            filter_metadata: Словарь фильтров по метаданным (точное совпадение).

        Returns:
            Список длиной Q, каждый элемент — список dict с ключами 'score' и 'metadata'.
        """
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
                for d, i in zip(dist_row, idx_row)  # noqa
                if i != -1
            ]
            for dist_row, idx_row in zip(distances, indices)  # noqa
        ]

    def _search_with_filter(
        self,
        prepared_queries: np.ndarray,
        top_k: int,
        filter_metadata: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        """Итеративный over-fetch: удваивает fetch_k пока не наберём top_k или не упрёмся в ntotal."""
        multiplier = self.filter_fetch_multiplier
        max_multiplier = self.filter_max_fetch_multiplier

        # Для каждого запроса собираем результаты отдельно, т.к. у разных запросов
        # может быть разный прогресс набора top_k кандидатов.
        n_queries = len(prepared_queries)
        batch_results: list[list[dict[str, Any]]] = [[] for _ in range(n_queries)]
        done = [False] * n_queries  # маркер «уже набрали top_k»

        while multiplier <= max_multiplier:
            fetch_k = min(top_k * multiplier, self.index.ntotal)
            distances, indices = self.index.search(prepared_queries, fetch_k)

            all_done = True
            for q_idx, (dist_row, idx_row) in enumerate(zip(distances, indices)):  # noqa
                if done[q_idx]:
                    continue

                row_res: list[dict[str, Any]] = []
                for d, i in zip(dist_row, idx_row):  # noqa
                    if i == -1:
                        continue
                    if self._match_filters(self.metadata[i], filter_metadata):
                        row_res.append({"score": float(d), "metadata": self.metadata[i]})
                    if len(row_res) == top_k:
                        break

                batch_results[q_idx] = row_res

                if len(row_res) >= top_k or fetch_k >= self.index.ntotal:
                    done[q_idx] = True
                else:
                    all_done = False

            if all_done:
                break

            multiplier *= 2
            logger.debug(
                "Фильтрация: не набрали top_k=%d для всех запросов, "
                "расширяем fetch до multiplier=%d",
                top_k,
                multiplier,
            )

        return batch_results

    # ------------------------------------------------------------------
    # Персистентность
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Сохраняет индекс и метаданные на диск.

        Создаёт два файла:
        - ``<directory>/index.faiss`` — FAISS-индекс.
        - ``<directory>/metadata.pkl`` — список словарей метаданных.

        Args:
            directory: Директория для сохранения.
        """
        self._check_consistency()
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_path = dir_path / "index.faiss"
        meta_path = dir_path / "metadata.pkl"

        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "VectorDB сохранена в '%s' (%d векторов).",
            directory,
            self.index.ntotal,
        )

    @classmethod
    def load(cls, directory: str | Path, **init_kwargs: Any) -> "FAISSVectorDB":
        """Загружает индекс и метаданные с диска.

        Args:
            directory: Директория, в которую ранее был вызван save().
            **init_kwargs: Параметры для __init__ (embedding_dim, index_type и т.д.).
                Должны совпадать с теми, что использовались при создании индекса.

        Returns:
            Новый инстанс FAISSVectorDB с восстановленным состоянием.

        Raises:
            FileNotFoundError: Если файлы индекса или метаданных не найдены.
        """
        dir_path = Path(directory)
        index_path = dir_path / "index.faiss"
        meta_path = dir_path / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Файл индекса не найден: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {meta_path}")

        instance = cls(**init_kwargs)
        instance.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            instance.metadata = pickle.load(f)

        instance._check_consistency()
        logger.info(
            "VectorDB загружена из '%s' (%d векторов).",
            directory,
            instance.index.ntotal,
        )
        return instance

    def reset(self) -> None:
        """Полностью очищает индекс и метаданные."""
        self.index.reset()
        self.metadata = []
        logger.info("VectorDB сброшена.")
