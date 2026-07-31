# src/rag_pipeline/retrieval/retriever.py
import logging
from typing import Any

from src.rag_pipeline.sdk.embedder import RAGInferenceEmbedder
from src.utils.vector_db import FAISSVectorDB


logger = logging.getLogger(__name__)


class BaseRetriever:
    """Продакшен-ретривер с поддержкой фильтрации и score thresholding.

    Принимает текстовый запрос (или список запросов), векторизует через
    ``RAGInferenceEmbedder`` и ищет в ``FAISSVectorDB``.
    """

    def __init__(
        self,
        embedder: RAGInferenceEmbedder,
        vector_db: FAISSVectorDB,
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
        """Ищет документы по одному текстовому запросу.

        Args:
            query: Текст запроса.
            top_k: Максимальное число возвращаемых документов.
            score_threshold: Минимальный порог косинусного сходства [0, 1].
                Документы с ``score < score_threshold`` отбрасываются после
                получения ``top_k`` кандидатов из FAISS. Фильтрация по порогу
                применяется поверх фильтрации по метаданным.
            filter_metadata: Словарь для точной фильтрации по полям метаданных.
                Делегируется в ``FAISSVectorDB.search``.

        Returns:
            Список dict с ключами ``'score'`` и ``'metadata'``,
            отсортированный по убыванию score. Может быть пустым.
        """
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
        """Ищет документы по списку запросов одним вызовом FAISS.

        Эффективнее N вызовов ``search`` при большом числе запросов —
        FAISS обрабатывает батч векторов за один проход индекса.

        Args:
            queries: Список текстовых запросов.
            top_k: Максимальное число документов на запрос.
            score_threshold: Минимальный порог score (применяется после FAISS).
            filter_metadata: Словарь фильтров по метаданным.

        Returns:
            Список длиной ``len(queries)``, каждый элемент — список результатов
            для соответствующего запроса.

        Raises:
            ValueError: Если ``queries`` пустой список.
        """
        if not queries:
            raise ValueError("queries не может быть пустым списком.")

        if self.vector_db.index.ntotal == 0:
            logger.warning(
                "Векторная БД пуста — поиск невозможен. Запустите индексацию перед поиском."
            )
            return [[] for _ in queries]

        # Векторизуем все запросы одним батчем
        query_vectors = self.embedder.encode(queries)

        # Один вызов FAISS для всего батча запросов
        raw_results = self.vector_db.search(
            query_vectors,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        if score_threshold is None:
            return raw_results

        # Применяем score_threshold поверх результатов FAISS
        return [
            [res for res in res_list if res["score"] >= score_threshold] for res_list in raw_results
        ]
