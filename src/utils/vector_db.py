import logging
from typing import Any

import faiss
import numpy as np


logger = logging.getLogger(__name__)


class FAISSVectorDB:
    """Локальная векторная база данных с поддержкой HNSW и пост-фильтрации."""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()

        if self.index_type == "hnsw":
            logger.info(
                "Инициализация FAISS: IndexHNSWFlat (M=%d, ef_c=%d, ef_s=%d)",
                m,
                ef_construction,
                ef_search,
            )
            # faiss.METRIC_INNER_PRODUCT эквивалентен косинусной близости для нормализованных векторов
            self.index = faiss.IndexHNSWFlat(embedding_dim, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        elif self.index_type == "flat":
            logger.info("Инициализация FAISS: IndexFlatIP (точный поиск)")
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"Неизвестный тип индекса: {self.index_type}")

        self.metadata: list[dict[str, Any]] = []

    def insert(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Размерность векторов {embeddings.shape[1]} не совпадает с embedding_dim {self.embedding_dim}"
            )

        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def _match_filters(self, doc_meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Проверяет, соответствует ли документ заданным фильтрам."""
        for key, value in filters.items():
            if doc_meta.get(key) != value:
                return False
        return True

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        # Если есть фильтры, запрашиваем из базы в 5 раз больше соседей (over-fetching),
        # чтобы после ручной фильтрации осталось нужное количество top_k.
        fetch_k = top_k * 5 if filter_metadata else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        distances, indices = self.index.search(query_embeddings.astype(np.float32), fetch_k)

        batch_results = []
        for dist_row, idx_row in zip(distances, indices):
            row_res = []
            for d, i in zip(dist_row, idx_row):
                if i == -1:
                    continue

                doc_meta = self.metadata[i]

                # Применяем фильтр
                if filter_metadata and not self._match_filters(doc_meta, filter_metadata):
                    continue

                row_res.append({"score": float(d), "metadata": doc_meta})

                if len(row_res) == top_k:
                    break

            batch_results.append(row_res)

        return batch_results

    def reset(self) -> None:
        self.index.reset()
        self.metadata = []
