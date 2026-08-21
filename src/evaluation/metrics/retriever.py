# src/evaluation/metrics/retriever.py
"""Метрики качества ретривера.

Поддерживает два сценария:
1. **Обучение** (``RetrievalEvaluationCallback``) — только bi-encoder метрики,
   реранкер на этом этапе не используется.
2. **Эталонный датасет** (``RetrieverEvaluator``) — полный двухэтапный пайплайн:
   сначала метрики bi-encoder (recall до реранкинга), затем метрики cross-encoder
   (NDCG/MRR/Precision по финальной выдаче после реранкинга).

Почему важно считать метрики на двух этапах:
    Recall@top_k (bi-encoder) — показывает, «не потерял» ли ретривер релевантные
    документы ещё до реранкинга. Если recall падает здесь — проблема в энкодере
    или в векторной базе, реранкер уже не поможет.

    NDCG@rerank_top_k (cross-encoder) — показывает качество финальной выдачи
    которую видит пользователь/LLM. Если recall до реранка высокий, но NDCG
    финальный низкий — проблема в реранкере или в том, что он ранжирует документы
    неверно.

    Без разделения этих двух этапов невозможно понять где деградация.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class RetrieverMetrics:
    """Индустриальные и бизнес-метрики качества ретривера.

    Метрики bi-encoder (считаются по кандидатам ДО реранкинга):
        recall_N:   доля запросов, где хотя бы один релевантный документ
                    попал в топ-N кандидатов bi-encoder.
        fnr:        False Negative Rate = 1 - recall_N.

    Метрики cross-encoder (считаются по финальной выдаче ПОСЛЕ реранкинга):
        mrr:        Mean Reciprocal Rank по rerank_top_k.
        precision_K: P@K по финальному срезу.
        ndcg_K:     nDCG@K по финальному срезу.

    Бизнес-метрики:
        empty_retrieval_rate: доля запросов с пустой выдачей (после score_threshold).
        ood_rate:             доля запросов где top-score bi-encoder < ood_threshold.
        redundancy_score:     среднее попарное сходство Жаккара в финальной выдаче.
                              Высокое значение -> дубликаты занимают окно контекста.

    Args:
        retrieval_top_k:  глубина первичного поиска (bi-encoder кандидаты).
        rerank_top_k:     глубина финальной выдачи (после реранкинга).
        similarity_threshold: минимальный dense score для фильтрации.
        ood_threshold:    порог OOD-детекции по top dense score.
    """

    def __init__(
        self,
        retrieval_top_k: int,
        rerank_top_k: int | None = None,
        similarity_threshold: float = 0.0,
        ood_threshold: float | None = None,
    ) -> None:
        self.retrieval_top_k = retrieval_top_k
        # Если реранкера нет — финальный срез = весь retrieval
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else retrieval_top_k
        self.similarity_threshold = similarity_threshold
        self.ood_threshold = ood_threshold

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def _compute_redundancy(self, results: list[dict[str, Any]]) -> float:
        if len(results) < 2:
            return 0.0
        texts = [r.get("metadata", {}).get("text", "") for r in results]
        sims = [self._jaccard_similarity(t1, t2) for t1, t2 in itertools.combinations(texts, 2)]
        return float(np.mean(sims)) if sims else 0.0

    def _ir_metrics(
        self,
        results: list[dict[str, Any]],
        true_ids: set,
        score_key: str,
    ) -> tuple[float, float, float, float]:
        """Считает MRR, Precision, nDCG, hits для одного запроса.

        Args:
            results:   список документов (уже отсортированных и срезанных).
            true_ids:  множество релевантных doc_id.
            score_key: ключ для сортировки — "score" (bi-encoder) или
                       "cross_encoder_score" (cross-encoder).

        Returns:
            (mrr_contrib, precision, ndcg, hits_count)
        """
        n_relevant = len(true_ids)
        hits = 0
        dcg = 0.0
        first_hit_rank: int | None = None

        for rank, res in enumerate(results):
            doc_id = res.get("metadata", {}).get("doc_id")
            if doc_id in true_ids:
                hits += 1
                dcg += 1.0 / np.log2(rank + 2)
                if first_hit_rank is None:
                    first_hit_rank = rank + 1

        mrr_contrib = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        precision = hits / len(results) if results else 0.0
        i_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_relevant, len(results))))
        ndcg = (dcg / i_dcg) if i_dcg > 0.0 else 0.0

        return mrr_contrib, precision, ndcg, hits

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def compute(
        self,
        search_results: list[list[dict[str, Any]]],
        ground_truth_ids: list[list[Any]],
    ) -> dict[str, float]:
        """Считает полный набор метрик: bi-encoder + cross-encoder + бизнес.

        Структура search_results[i]:
            Список документов в порядке финальной выдачи (после реранкинга,
            если он применялся). Каждый документ содержит:
                - "score":               dense score от bi-encoder (всегда есть).
                - "cross_encoder_score": score от cross-encoder (есть если был реранкинг).
                - "metadata": {"doc_id": ..., "text": ...}

            Важно: список должен содержать ВСЕ кандидаты (top retrieval_top_k),
            отсортированные по финальному score. Метод сам делает срез до
            rerank_top_k для cross-encoder метрик и использует весь список
            для bi-encoder recall.
        """
        n_queries = len(ground_truth_ids)

        if n_queries == 0:
            return self._empty_metrics()

        if len(search_results) != n_queries:
            raise ValueError(
                f"Длина search_results ({len(search_results)}) "
                f"не совпадает с ground_truth_ids ({n_queries})."
            )

        # Накопители — bi-encoder (до реранкинга)
        bi_recall_sum = 0.0
        ood_count = 0

        # Накопители — cross-encoder / финальная выдача
        ce_mrr_sum = 0.0
        ce_precision_sum = 0.0
        ce_ndcg_sum = 0.0
        ce_recall_sum = 0.0
        empty_count = 0
        redundancy_sum = 0.0

        for res_list, gt_ids in zip(search_results, ground_truth_ids):
            true_ids = set(gt_ids)

            # ── OOD-детекция по top dense score (bi-encoder, до всяких фильтров) ──
            if self.ood_threshold is not None:
                top_dense_score = res_list[0].get("score", 0.0) if res_list else 0.0
                if top_dense_score < self.ood_threshold:
                    ood_count += 1

            # ── Фильтрация по similarity_threshold (по dense score) ──
            filtered = [
                r for r in res_list
                if r.get("score", 1.0) >= self.similarity_threshold
            ]

            if not filtered:
                empty_count += 1
                continue

            # ── Bi-encoder recall: был ли релевантный документ среди всех кандидатов? ──
            # Используем весь filtered (retrieval_top_k), а не срез rerank_top_k,
            # потому что нас интересует не упустил ли энкодер документ вообще.
            bi_candidates = filtered[:self.retrieval_top_k]
            bi_doc_ids = {r.get("metadata", {}).get("doc_id") for r in bi_candidates}
            bi_recall_sum += 1.0 if (true_ids & bi_doc_ids) else 0.0

            # ── Cross-encoder / финальная выдача: срез до rerank_top_k ──
            # Если реранкер применялся — документы уже отсортированы по cross_encoder_score.
            # Если нет — порядок из bi-encoder, cross_encoder_score отсутствует.
            final_results = filtered[:self.rerank_top_k]

            has_reranker = any("cross_encoder_score" in r for r in final_results)
            score_key = "cross_encoder_score" if has_reranker else "score"

            mrr_c, prec, ndcg, hits = self._ir_metrics(final_results, true_ids, score_key)
            ce_mrr_sum += mrr_c
            ce_precision_sum += prec
            ce_ndcg_sum += ndcg
            ce_recall_sum += 1.0 if hits > 0 else 0.0

            redundancy_sum += self._compute_redundancy(final_results)

        k_ret = self.retrieval_top_k
        k_rank = self.rerank_top_k

        bi_recall = bi_recall_sum / n_queries
        ce_recall = ce_recall_sum / n_queries

        return {
            # Bi-encoder: насколько хорошо энкодер «не потерял» релевантные документы
            f"recall_{k_ret}_biencoder": bi_recall,
            f"fnr_{k_ret}_biencoder": 1.0 - bi_recall,

            # Cross-encoder / финальная выдача: качество того, что видит пользователь
            f"recall_{k_rank}_final": ce_recall,
            "mrr": ce_mrr_sum / n_queries,
            f"precision_{k_rank}": ce_precision_sum / n_queries,
            f"ndcg_{k_rank}": ce_ndcg_sum / n_queries,

            # Бизнес-метрики
            "empty_retrieval_rate": empty_count / n_queries,
            "ood_rate": ood_count / n_queries,
            "redundancy_score": redundancy_sum / n_queries,
        }

    def _empty_metrics(self) -> dict[str, float]:
        k_ret = self.retrieval_top_k
        k_rank = self.rerank_top_k
        return {
            f"recall_{k_ret}_biencoder": 0.0,
            f"fnr_{k_ret}_biencoder": 1.0,
            f"recall_{k_rank}_final": 0.0,
            "mrr": 0.0,
            f"precision_{k_rank}": 0.0,
            f"ndcg_{k_rank}": 0.0,
            "empty_retrieval_rate": 1.0,
            "ood_rate": 1.0,
            "redundancy_score": 0.0,
        }