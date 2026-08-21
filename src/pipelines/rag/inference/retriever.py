# src/pipelines/rag/inference/retriever.py
import logging
import time
from typing import Any

from src.pipelines.rag.api.metrics import (
    RAG_ENCODE_DURATION_SECONDS,
    RAG_RERANK_DURATION_SECONDS,
    RAG_SEARCH_DURATION_SECONDS,
)
from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder
from src.pipelines.rag.inference.reranker import CrossEncoderReranker
from src.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Оркестратор гибридного поиска (Dense + Sparse/BM25) с опциональным реранкингом.

    Пайплайн:
        1. encode(queries)          -> query_vectors
        2. vector_db.search_hybrid  -> top-(top_k * rerank_factor) кандидатов
        3. [опц.] reranker.rerank   -> пересортированные top_k документов

    rerank_factor контролирует компромисс качество/скорость:
        При rerank_factor=3 и top_k=5 ретривер сначала забирает 15 кандидатов
        из Qdrant, реранкер из них выбирает лучшие 5.
        Увеличение factor -> лучше recall, но дороже реранкинг.

    Компонентные замеры:
        После каждого batch_search() заполняется self.last_batch_timing_sec —
        словарь с ключами "encode", "search", "rerank" (секунды, суммарно по батчу).
        RetrieverEvaluator читает его для per-query latency метрик.
        Параллельно те же значения пишутся в Prometheus Histogram'ы.

        source="batch" — метки для Prometheus, отделяют batch_search() от
        одиночных search() вызовов через API (там source="rest" пишет search.py).
    """

    def __init__(
        self,
        embedder: RAGInferenceEmbedder,
        vector_db: BaseVectorStore,
        reranker: CrossEncoderReranker | None = None,
        rerank_factor: int = 3,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.vector_db = vector_db
        self.reranker = reranker
        self.rerank_factor = rerank_factor
        self.top_k = top_k

        self.last_batch_timing_sec: dict[str, float] = {}

        if reranker is not None:
            logger.info(
                "HybridRetriever: реранкер включён (rerank_factor=%d).", rerank_factor
            )
        else:
            logger.info("HybridRetriever: реранкер отключён.")

    def search(
        self,
        query: str,
        top_k: int | None = None,  # <--- Меняем на None
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        actual_top_k = top_k if top_k is not None else self.top_k
        results = self.batch_search(
            queries=[query],
            top_k=actual_top_k,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )
        return results[0]

    def batch_search(
        self,
        queries: list[str],
        top_k: int | None = None,  # <--- Меняем на None
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        actual_top_k = top_k if top_k is not None else self.top_k

        if not queries:
            raise ValueError("queries не может быть пустым списком.")

        if self.vector_db.ntotal == 0:
            logger.warning("Векторная БД пуста — поиск невозможен.")
            self.last_batch_timing_sec = {}
            return [[] for _ in queries]

        timing: dict[str, float] = {}

        # --- Этап 1: векторизация ---
        t0 = time.perf_counter()
        try:
            query_vectors = self.embedder.encode(queries)
        except RuntimeError as e:
            logger.error("Ошибка векторизации: %s", e)
            self.last_batch_timing_sec = {}
            return [[] for _ in queries]
        encode_sec = time.perf_counter() - t0
        timing["encode"] = encode_sec
        RAG_ENCODE_DURATION_SECONDS.labels(source="batch").observe(encode_sec)

        # --- Этап 2: гибридный поиск ---
        # Если реранкер включён — забираем больше кандидатов, потом срезаем до top_k.
        fetch_k = actual_top_k * self.rerank_factor if self.reranker is not None else actual_top_k

        t0 = time.perf_counter()
        raw_results = self.vector_db.search_hybrid(
            query_vectors=query_vectors,
            query_texts=queries,
            top_k=fetch_k,
            filter_metadata=filter_metadata,
        )
        search_sec = time.perf_counter() - t0
        timing["search"] = search_sec
        RAG_SEARCH_DURATION_SECONDS.labels(source="batch").observe(search_sec)

        # --- Этап 3: score_threshold (до реранкинга — по dense score) ---
        if score_threshold is not None:
            raw_results = [
                [res for res in res_list if res["score"] >= score_threshold]
                for res_list in raw_results
            ]

        # --- Этап 4: реранкинг (опциональный) ---
        if self.reranker is None:
            self.last_batch_timing_sec = timing
            return raw_results

        t0 = time.perf_counter()
        reranked = []
        for query_text, candidates in zip(queries, raw_results):
            if not candidates:
                reranked.append([])
                continue
            reranked.append(
                self.reranker.rerank(query=query_text, documents=candidates, top_k=actual_top_k)
            )
        rerank_sec = time.perf_counter() - t0
        timing["rerank"] = rerank_sec
        RAG_RERANK_DURATION_SECONDS.labels(source="batch").observe(rerank_sec)

        self.last_batch_timing_sec = timing
        return reranked