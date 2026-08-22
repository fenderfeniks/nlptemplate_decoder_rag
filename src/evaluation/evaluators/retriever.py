# src/evaluation/evaluators/retriever.py
"""Оркестратор инференса и подсчёта метрик для RAG-ретривера.

Инкапсулирует пакетный поиск, замеры производительности (Latency),
оценку стоимости (Context Tokens) и проверку бизнес-SLA (Drift Check).

Архитектура двухэтапной оценки:
    batch_search возвращает кандидатов в финальном порядке (bi-encoder или
    bi-encoder + cross-encoder). RetrieverMetrics.compute видит весь список
    и сама делает срез до retrieval_top_k для recall-bi-encoder и до
    rerank_top_k для финальных IR-метрик — таким образом мы не теряем
    информацию о том, что реально нашёл bi-encoder.

    Latency разбита на три независимых замера:
        encode_latency_ms:  только инференс энкодера (векторизация).
        search_latency_ms:  только поиск в векторной базе (без энкодера).
        rerank_latency_ms:  только инференс реранкера (если включён).
        total_latency_ms:   весь пайплайн от запроса до ответа.
    Без раздельных замеров нельзя понять где именно деградация при SLA-алерте.
"""

import logging
import sys
import time
from typing import Any

from src.evaluation.metrics.retriever import RetrieverMetrics
from src.utils.logging.protocol import ExperimentLogger


logger = logging.getLogger(__name__)


class RetrieverEvaluator:
    """Оркестратор оценки ретривера и проверки бизнес-метрик.

    Args:
        retriever:        Собранный пайплайн поиска (HybridRetriever).
        tokenizer:        Токенизатор для подсчёта реального размера контекста.
        retrieval_top_k:  Глубина первичного поиска (bi-encoder кандидаты).
        rerank_top_k:     Глубина финальной выдачи (срез после реранкинга).
        ood_threshold:    Порог для вычисления OOD-запросов по dense score.
        drift_cfg:        Конфиг порогов для бизнес-метрик. Поддерживаемые ключи:
                              max_latency_ms       — SLA по total_latency_ms_per_query.
                              max_context_tokens   — бюджет токенов контекста на запрос.
                              min_ndcg             — минимальный ndcg_{rerank_top_k}.
                              min_recall_biencoder — минимальный recall_{retrieval_top_k}_biencoder.
        raise_on_drift:   Если True — бросает RuntimeError при дрифте вместо sys.exit(1).
                          Используй True в Lightning-callback (обучение), False в eval-скрипте.
    """

    def __init__(
        self,
        retriever: Any,
        tokenizer: Any,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
        ood_threshold: float | None = 0.3,
        drift_cfg: dict[str, Any] | None = None,
        raise_on_drift: bool = False,
    ) -> None:
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.raise_on_drift = raise_on_drift

        self.metrics_calculator = RetrieverMetrics(
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            ood_threshold=ood_threshold,
        )
        self.drift_cfg = drift_cfg or {}

    # ------------------------------------------------------------------
    # Drift-check
    # ------------------------------------------------------------------

    def _check_drift(self, metrics: dict[str, float]) -> None:
        """Проверяет метрики на соответствие заданным SLA и бюджетам.

        При drift_detected:
            raise_on_drift=False -> sys.exit(1)  (eval-скрипт, CI/CD)
            raise_on_drift=True  -> RuntimeError  (Lightning callback — не убиваем процесс)
        """
        drift_detected = False
        messages: list[str] = []

        # 1. SLA по суммарной задержке
        max_latency = self.drift_cfg.get("max_latency_ms")
        if max_latency and metrics.get("total_latency_ms_per_query", 0) > max_latency:
            msg = (
                f"ДРИФТ (SLA): total_latency {metrics['total_latency_ms_per_query']:.1f} ms "
                f"превышает допустимые {max_latency} ms."
            )
            logger.error(msg)
            messages.append(msg)
            drift_detected = True

        # 2. Бюджет токенов контекста
        max_tokens = self.drift_cfg.get("max_context_tokens")
        if max_tokens and metrics.get("avg_context_tokens", 0) > max_tokens:
            msg = (
                f"ДРИФТ (Бюджет): avg_context_tokens {metrics['avg_context_tokens']:.1f} "
                f"превышает лимит {max_tokens}."
            )
            logger.error(msg)
            messages.append(msg)
            drift_detected = True

        # 3. Качество финальной выдачи (cross-encoder)
        min_ndcg = self.drift_cfg.get("min_ndcg")
        ndcg_key = f"ndcg_{self.rerank_top_k}"
        if min_ndcg and metrics.get(ndcg_key, 0.0) < min_ndcg:
            msg = (
                f"ДРИФТ (Качество): {ndcg_key}={metrics[ndcg_key]:.4f} упал ниже порога {min_ndcg}."
            )
            logger.error(msg)
            messages.append(msg)
            drift_detected = True

        # 4. Recall bi-encoder — независимый алерт, не скрытый за NDCG
        min_recall_bi = self.drift_cfg.get("min_recall_biencoder")
        recall_bi_key = f"recall_{self.retrieval_top_k}_biencoder"
        if min_recall_bi and metrics.get(recall_bi_key, 0.0) < min_recall_bi:
            msg = (
                f"ДРИФТ (Bi-encoder): {recall_bi_key}={metrics[recall_bi_key]:.4f} "
                f"упал ниже порога {min_recall_bi}. "
                f"Проблема в энкодере или индексе — реранкер не поможет."
            )
            logger.error(msg)
            messages.append(msg)
            drift_detected = True

        if drift_detected:
            summary = "Бизнес-метрики не прошли проверку (Drift Detected): " + "; ".join(messages)
            logger.critical(summary)
            if self.raise_on_drift:
                raise RuntimeError(summary)
            sys.exit(1)
        else:
            logger.info("Проверка дрифта пройдена: SLA и бюджеты в норме.")

    # ------------------------------------------------------------------
    # Основной метод оценки
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queries: list[str],
        ground_truths: list[list[Any]],
        metrics_logger: ExperimentLogger,
        stage: str = "test",
        global_step: int = 0,
    ) -> dict[str, float]:
        """Запускает полный цикл оценки и возвращает словарь метрик.

        Returns:
            Словарь с метриками bi-encoder, cross-encoder, latency, токены, бизнес.
        """
        if not queries:
            logger.warning("Пустой список запросов, оценка пропущена.")
            return {}

        n = len(queries)
        logger.info(
            "Запуск поиска по %d запросам (retrieval_top_k=%d, rerank_top_k=%d)...",
            n,
            self.retrieval_top_k,
            self.rerank_top_k,
        )

        # ── 1. Batch search + замер total latency ──
        t_total_start = time.perf_counter()
        search_results = self.retriever.batch_search(
            queries=queries,
            top_k=self.retrieval_top_k,
            # rerank_top_k НЕ передаём — HybridRetriever управляет этим через rerank_factor.
            # Финальный срез до rerank_top_k делает RetrieverMetrics.
        )
        total_latency_sec = time.perf_counter() - t_total_start

        # ── 2. Компонентная latency (если retriever пишет статистику) ──
        # HybridRetriever может опционально накапливать timing в self.last_timing_sec.
        # Если нет — пишем только total.
        timing = getattr(self.retriever, "last_batch_timing_sec", {})
        latency_metrics: dict[str, float] = {
            "total_latency_ms_per_query": (total_latency_sec / n) * 1000,
        }
        for component in ("encode", "search", "rerank"):
            if component in timing:
                latency_metrics[f"{component}_latency_ms_per_query"] = timing[component] / n * 1000

        # ── 3. IR-метрики (bi-encoder + cross-encoder) ──
        metrics = self.metrics_calculator.compute(search_results, ground_truths)

        # ── 4. Cost-метрики (context tokens по финальному срезу) ──
        total_tokens = 0
        for res_list in search_results:
            for res in res_list[: self.rerank_top_k]:
                text = res.get("metadata", {}).get("text", "")
                if text:
                    total_tokens += len(self.tokenizer.encode(text, add_special_tokens=False))
        metrics["avg_context_tokens"] = total_tokens / n

        # Объединяем все метрики
        metrics.update(latency_metrics)

        # ── 5. Логирование ──
        metrics_logger.log_metrics(metrics, stage=stage, step=global_step)
        for name, value in sorted(metrics.items()):
            logger.info("%s | %s: %.4f", stage.upper(), name, value)

        # ── 6. Drift-check ──
        self._check_drift(metrics)

        return metrics
