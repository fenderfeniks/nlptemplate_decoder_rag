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

    Пайплайн (нормальный режим):
        1. encode(queries)          -> query_vectors
        2. vector_db.search_hybrid  -> top-(top_k * rerank_factor) кандидатов
        3. [опц.] reranker.rerank   -> пересортированные top_k документов

    Fallback-цепочка при ошибках (рантайм):

        search_hybrid упал:
            → dense-only (vector_db.search_dense)
            → BM25-only  (vector_db.search_sparse) если dense тоже упал
            → [] если все три упали

        reranker упал:
            → возвращаем верхние top_k кандидатов по dense/hybrid score как есть.
              Количество совпадает с ожидаемым после реранкинга — клиент не видит разницы
              в размере ответа, только теряет качество сортировки.

    Деградация фиксируется в self.last_batch_degradation — словарь вида:
        {"search": "dense_only" | "sparse_only" | "failed", "rerank": "skipped"}
    search.py читает его для метрик (при необходимости).

    rerank_factor контролирует компромисс качество/скорость:
        При rerank_factor=3 и top_k=5 ретривер сначала забирает 15 кандидатов
        из Qdrant, реранкер из них выбирает лучшие 5.
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
        self.last_batch_degradation: dict[str, str] = {}

        if reranker is not None:
            logger.info("HybridRetriever: реранкер включён (rerank_factor=%d).", rerank_factor)
        else:
            logger.info("HybridRetriever: реранкер отключён.")

    def search(
        self,
        query: str,
        top_k: int | None = None,
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
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        actual_top_k = top_k if top_k is not None else self.top_k
        self.last_batch_degradation = {}

        if not queries:
            raise ValueError("queries не может быть пустым списком.")

        if self.vector_db.ntotal == 0:
            logger.warning("Векторная БД пуста — поиск невозможен.")
            self.last_batch_timing_sec = {}
            return [[] for _ in queries]

        timing: dict[str, float] = {}

        # --- Этап 1: векторизация ---
        # Encode нужен для dense и hybrid. При падении encode дальше идти некуда —
        # без векторов ни dense, ни hybrid не работают. Падаем сразу на BM25-only.
        query_vectors: list | None = None
        t0 = time.perf_counter()
        try:
            query_vectors = self.embedder.encode(queries)
            encode_sec = time.perf_counter() - t0
            timing["encode"] = encode_sec
            RAG_ENCODE_DURATION_SECONDS.labels(source="batch").observe(encode_sec)
        except RuntimeError as e:
            encode_sec = time.perf_counter() - t0
            timing["encode"] = encode_sec
            logger.error("Ошибка векторизации — переходим сразу к BM25-only fallback: %s", e)
            # query_vectors остаётся None → search_hybrid и dense пропускаем

        # --- Этап 2: поиск с fallback-цепочкой ---
        fetch_k = actual_top_k * self.rerank_factor if self.reranker is not None else actual_top_k
        raw_results, search_mode = self._search_with_fallback(
            queries=queries,
            query_vectors=query_vectors,
            fetch_k=fetch_k,
            filter_metadata=filter_metadata,
            timing=timing,
        )

        if search_mode != "hybrid":
            self.last_batch_degradation["search"] = search_mode

        # --- Этап 3: score_threshold (до реранкинга — по dense score) ---
        if score_threshold is not None:
            raw_results = [
                [res for res in res_list if res["score"] >= score_threshold]
                for res_list in raw_results
            ]

        # --- Этап 4: реранкинг с fallback ---
        if self.reranker is None:
            self.last_batch_timing_sec = timing
            return raw_results

        reranked, rerank_degraded = self._rerank_with_fallback(
            queries=queries,
            raw_results=raw_results,
            actual_top_k=actual_top_k,
            timing=timing,
        )

        if rerank_degraded:
            self.last_batch_degradation["rerank"] = "skipped"

        self.last_batch_timing_sec = timing
        return reranked

    # ------------------------------------------------------------------
    # Внутренние методы fallback
    # ------------------------------------------------------------------

    def _search_with_fallback(
        self,
        queries: list[str],
        query_vectors: list | None,
        fetch_k: int,
        filter_metadata: dict[str, Any] | None,
        timing: dict[str, float],
    ) -> tuple[list[list[dict[str, Any]]], str]:
        """Гибридный поиск с fallback: hybrid → dense → BM25 → [].

        Returns:
            (results, mode) где mode — фактически использованный метод поиска:
            "hybrid" | "dense_only" | "sparse_only" | "failed"
        """
        empty = [[] for _ in queries]

        # --- Попытка 1: hybrid (dense + BM25 + RRF) ---
        if query_vectors is not None:
            t0 = time.perf_counter()
            try:
                results = self.vector_db.search_hybrid(
                    query_vectors=query_vectors,
                    query_texts=queries,
                    top_k=fetch_k,
                    filter_metadata=filter_metadata,
                )
                timing["search"] = time.perf_counter() - t0
                RAG_SEARCH_DURATION_SECONDS.labels(source="batch").observe(timing["search"])
                return results, "hybrid"
            except Exception as e:
                timing["search"] = time.perf_counter() - t0
                logger.warning(
                    "search_hybrid упал (%s: %s) — пробуем dense-only fallback.",
                    type(e).__name__,
                    e,
                )

        # --- Попытка 2: dense-only ---
        if query_vectors is not None:
            t0 = time.perf_counter()
            try:
                results = self.vector_db.search_dense(
                    query_vectors=query_vectors,
                    top_k=fetch_k,
                    filter_metadata=filter_metadata,
                )
                timing["search_dense_fallback"] = time.perf_counter() - t0
                logger.warning("Используем dense-only fallback.")
                return results, "dense_only"
            except Exception as e:
                timing["search_dense_fallback"] = time.perf_counter() - t0
                logger.warning(
                    "dense-only тоже упал (%s: %s) — пробуем BM25-only fallback.",
                    type(e).__name__,
                    e,
                )

        # --- Попытка 3: BM25-only (sparse) ---
        t0 = time.perf_counter()
        try:
            results = self.vector_db.search_sparse(
                query_texts=queries,
                top_k=fetch_k,
                filter_metadata=filter_metadata,
            )
            timing["search_sparse_fallback"] = time.perf_counter() - t0
            logger.warning("Используем BM25-only (sparse) fallback.")
            return results, "sparse_only"
        except Exception as e:
            timing["search_sparse_fallback"] = time.perf_counter() - t0
            logger.error(
                "Все методы поиска упали. BM25 (%s: %s). Возвращаем пустые результаты.",
                type(e).__name__,
                e,
            )
            return empty, "failed"

    def _rerank_with_fallback(
        self,
        queries: list[str],
        raw_results: list[list[dict[str, Any]]],
        actual_top_k: int,
        timing: dict[str, float],
    ) -> tuple[list[list[dict[str, Any]]], bool]:
        """Реранкинг с fallback: при ошибке возвращает top_k кандидатов по исходному score.

        Returns:
            (results, degraded) где degraded=True если реранкинг был пропущен.
        """
        t0 = time.perf_counter()
        try:
            reranked = []
            for query_text, candidates in zip(queries, raw_results, strict=True):
                if not candidates:
                    reranked.append([])
                    continue
                reranked.append(
                    self.reranker.rerank(
                        query=query_text,
                        documents=candidates,
                        top_k=actual_top_k,
                    )
                )
            rerank_sec = time.perf_counter() - t0
            timing["rerank"] = rerank_sec
            RAG_RERANK_DURATION_SECONDS.labels(source="batch").observe(rerank_sec)
            return reranked, False

        except Exception as e:
            timing["rerank_failed"] = time.perf_counter() - t0
            logger.error(
                "Реранкер упал в рантайме (%s: %s) — возвращаем top_%d кандидатов по исходному score.",
                type(e).__name__,
                e,
                actual_top_k,
            )
            # Срезаем до actual_top_k — клиент получает ровно столько,
            # сколько ожидал бы после реранкинга.
            fallback = [candidates[:actual_top_k] for candidates in raw_results]
            return fallback, True
