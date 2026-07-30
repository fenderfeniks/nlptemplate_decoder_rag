# src/rag_pipeline/retrieval/retriever.py
import logging
from typing import Any

from src.rag_pipeline.inference.embedder import RAGInferenceEmbedder
from src.utils.vector_db import FAISSVectorDB


logger = logging.getLogger(__name__)


class BaseRetriever:
    """Продакшен-ретривер с поддержкой фильтрации и score thresholding."""

    def __init__(
        self,
        embedder: RAGInferenceEmbedder,
        vector_db: FAISSVectorDB,
    ):
        self.embedder = embedder
        self.vector_db = vector_db

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ищет документы по запросу.

        Args:
            query: Текст запроса.
            top_k: Максимальное количество документов.
            score_threshold: Минимальный порог косинусной близости.
            filter_metadata: Словарь для фильтрации (поддерживается на уровне БД).
        """
        # 1. Векторизуем запрос через готовый эмбеддер
        query_vector = self.embedder.encode([query])

        # 2. Ищем в базе (допущение, что метод search БД поддерживает фильтры)
        raw_results = self.vector_db.search(
            query_vector, top_k=top_k, filter_metadata=filter_metadata
        )[0]  # Берем [0], так как запрос был один

        # 3. Фильтруем по порогу уверенности модели
        final_results = []
        for res in raw_results:
            if score_threshold is not None and res["score"] < score_threshold:
                continue
            final_results.append(res)

        return final_results
