import logging
from typing import Any

from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder
from src.vector_store.base import BaseVectorStore


logger = logging.getLogger(__name__)


class BaseRetriever:
    """Продакшен-ретривер с поддержкой фильтрации и score thresholding.

    Принимает текстовый запрос (или список запросов), векторизует через
    ``RAGInferenceEmbedder`` и ищет в ``BaseVectorStore``.
    """

    def __init__(
        self,
        embedder: RAGInferenceEmbedder,
        vector_db: BaseVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_db = vector_db

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет документы по одному текстовому запросу."""
        results = self.batch_search(
            queries=[query],
            top_k=top_k,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )
        return results[0]

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Ищет документы по списку запросов одним вызовом хранилища."""
        if not queries:
            raise ValueError("queries не может быть пустым списком.")

        # Обращаемся напрямую к ntotal согласно протоколу BaseVectorStore
        if self.vector_db.ntotal == 0:
            logger.warning(
                "Векторная БД пуста — поиск невозможен. Запустите индексацию перед поиском."
            )
            return [[] for _ in queries]

        try:
            query_vectors = self.embedder.encode(queries)
        except RuntimeError as e:
            logger.error(
                "Ошибка векторизации запросов (%d шт.): %s. Возвращаем пустые результаты.",
                len(queries),
                e,
            )
            return [[] for _ in queries]

        raw_results = self.vector_db.search(
            query_vectors,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        if score_threshold is None:
            return raw_results

        return [
            [res for res in res_list if res["score"] >= score_threshold] for res_list in raw_results
        ]
