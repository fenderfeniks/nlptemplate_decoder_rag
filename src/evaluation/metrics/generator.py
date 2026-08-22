# src/evaluation/metrics/generator.py
"""Метрики качества генерации текста (decoder / SFT).

Четыре уровня оценки:
    StatisticalMetrics      — ROUGE, BLEU (быстро, без LLM).
    RagasMetrics            — Ragas-метрики с LLM-судьёй (Answer Correctness, Faithfulness и др.).
    GenerationSpeedMetrics  — Бизнес-метрики: латентность, throughput, token counts.
                              Требует передачи ``generation_stats`` (заполняется DecoderEvaluator).
    TextQualityMetrics      — Бизнес-метрики качества текста без LLM: repetition rate,
                              diversity (unique n-gram ratio), length ratio, empty-response rate.
    GeneratorMetricsPipeline — оркестратор: прогоняет данные через все сконфигурированные
                               метрики и сливает результаты в один словарь.

Использование:
    Инстанцируется через Hydra из configs/evaluation/metrics/*.yaml.
    Передаётся в DecoderEvaluator как metrics_pipeline.

    GenerationSpeedMetrics получает ``generation_stats`` через ``compute(..., extra=...)``.
    DecoderEvaluator собирает stats в ``_generate_chunks_with_stats`` и прокидывает через
    ``GeneratorMetricsPipeline.compute_all(..., extra=...)``.

Важно про импорты:
    В этом модуле намеренно НЕ делается `import evaluate` на верхнем уровне,
    а `from ragas import evaluate as ragas_evaluate` — чтобы не было конфликта
    имён между HuggingFace evaluate и Ragas evaluate в одном модуле.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Protocol

import evaluate as hf_evaluate  # hf_evaluate — явный алиас, без конфликта с ragas


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Протокол метрики
# ------------------------------------------------------------------


class GenerativeMetric(Protocol):
    """Единый интерфейс для любой метрики генератора.

    ``extra`` — произвольный словарь с боковыми данными, которые нельзя
    вывести из текстов (например, ``generation_stats`` с замерами времени
    и токенов из ``DecoderEvaluator._generate_chunks_with_stats``).
    Метрики, которым extra не нужен, просто его игнорируют.
    """

    def compute(
        self,
        prompts: list[str],
        generated: list[str],
        targets: list[str],
        contexts: list[list[str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]: ...


# ------------------------------------------------------------------
# Пайплайн
# ------------------------------------------------------------------


class GeneratorMetricsPipeline:
    """Пайплайн последовательного вычисления всех метрик.

    Принимает список инстанцированных через Hydra метрик.
    Ошибка в одной метрике не роняет остальные — логируется и пропускается.

    Args:
        metrics: Список объектов, реализующих ``GenerativeMetric``.
    """

    def __init__(self, metrics: list[GenerativeMetric]) -> None:
        self.metrics = metrics

    def compute_all(
        self,
        prompts: list[str],
        generated: list[str],
        targets: list[str],
        contexts: list[list[str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Прогоняет данные через все метрики, сливает результаты.

        Args:
            extra: Произвольный словарь с боковыми данными (generation_stats и т.п.).
                   Прокидывается в каждую метрику — метрики, которым он не нужен,
                   принимают **kwargs или явно его игнорируют.
        """
        results: dict[str, float] = {}
        for metric in self.metrics:
            cls_name = metric.__class__.__name__
            try:
                partial = metric.compute(prompts, generated, targets, contexts, extra)
                results.update(partial)
                logger.debug("%s: %s", cls_name, partial)
            except Exception as e:
                logger.error("Метрика %s упала: %s — пропускаем.", cls_name, e)
        return results


# ------------------------------------------------------------------
# StatisticalMetrics — ROUGE + BLEU
# ------------------------------------------------------------------


class StatisticalMetrics:
    """ROUGE и BLEU через HuggingFace ``evaluate``.

    Быстрые референсные метрики без LLM. Подходят для всех этапов
    обучения (каждая val-эпоха). ROUGE-L хорошо коррелирует с качеством
    для задач суммаризации и перефразирования; BLEU — для переводов.

    Args:
        use_rouge: Считать ли ROUGE (rouge1 + rougeL). По умолчанию True.
        use_bleu: Считать ли SacreBLEU. По умолчанию True.
    """

    def __init__(self, use_rouge: bool = True, use_bleu: bool = True) -> None:
        # Ленивая инициализация — не грузим модели при импорте модуля
        self._rouge = hf_evaluate.load("rouge") if use_rouge else None
        self._bleu = hf_evaluate.load("sacrebleu") if use_bleu else None

    def compute(
        self,
        prompts: list[str],
        generated: list[str],
        targets: list[str],
        contexts: list[list[str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        results: dict[str, float] = {}

        if self._rouge:
            rouge_res = self._rouge.compute(
                predictions=generated,
                references=targets,
                use_stemmer=True,
            )
            results["rouge1"] = self._extract_score(rouge_res["rouge1"])
            results["rougeL"] = self._extract_score(rouge_res["rougeL"])

        if self._bleu:
            bleu_res = self._bleu.compute(
                predictions=generated,
                references=[[t] for t in targets],
            )
            results["bleu"] = float(bleu_res["score"])

        return results

    @staticmethod
    def _extract_score(score: Any) -> float:
        """Нормализует разные форматы вывода ROUGE в float."""
        if hasattr(score, "mid"):
            # Старый формат datasets<2.x: AggregateScore с .mid.fmeasure
            return float(score.mid.fmeasure)
        if isinstance(score, (list, tuple)) and score:
            return float(score[0])
        return float(score)


# ------------------------------------------------------------------
# RagasMetrics — LLM-as-a-judge
# ------------------------------------------------------------------


class RagasMetrics:
    """Ragas-метрики с LLM-судьёй (Answer Correctness, Faithfulness и др.).

    Ленивая инициализация: ragas-объекты создаются при первом вызове
    ``compute``, чтобы не грузить LLM-судью во время обучения.

    Если ``contexts`` не передан — заполняется пустыми строками с
    предупреждением. Метрики, зависящие от контекста (Faithfulness,
    ContextPrecision), в этом случае могут давать некорректные значения.

    Если нужны локальные LLM-судьи вместо OpenAI — передайте
    ``llm`` и ``embeddings`` в конструктор и прокиньте в ``ragas.evaluate``.

    Args:
        metrics_to_run: Список строковых ключей метрик Ragas.
            Поддерживаемые: ``"answer_correctness"``, ``"faithfulness"``,
            ``"answer_relevancy"``, ``"context_precision"``.
        raise_exceptions: Пробрасывать ли исключения из LLM-судьи.
            ``False`` — не роняет пайплайн при разовом сбое судьи.
    """

    _METRICS_MAP = {
        "answer_correctness": "ragas.metrics.answer_correctness",
        "faithfulness": "ragas.metrics.faithfulness",
        "answer_relevancy": "ragas.metrics.answer_relevancy",
        "context_precision": "ragas.metrics.context_precision",
    }

    def __init__(
        self,
        metrics_to_run: list[str],
        raise_exceptions: bool = False,
    ) -> None:
        self.metrics_to_run = metrics_to_run
        self.raise_exceptions = raise_exceptions
        self._metric_objects: list[Any] = []
        self._initialized = False

    def _initialize(self) -> None:
        """Ленивая инициализация ragas-объектов при первом вызове compute."""
        if self._initialized:
            return
        self._initialized = True

        # Импортируем ragas-метрики по имени чтобы избежать загрузки LLM при старте
        import importlib

        for key in self.metrics_to_run:
            if key not in self._METRICS_MAP:
                logger.warning("RagasMetrics: '%s' не в маппинге — пропускаем.", key)
                continue
            module_path, attr = self._METRICS_MAP[key].rsplit(".", 1)
            try:
                mod = importlib.import_module(module_path)
                self._metric_objects.append(getattr(mod, attr))
            except Exception as e:
                logger.error("RagasMetrics: не удалось загрузить '%s': %s", key, e)

    def compute(
        self,
        prompts: list[str],
        generated: list[str],
        targets: list[str],
        contexts: list[list[str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        self._initialize()

        if not self._metric_objects:
            logger.warning("RagasMetrics: нет доступных метрик.")
            return {}

        metrics_to_run = self._metric_objects

        if contexts is None:
            logger.warning(
                "RagasMetrics: 'contexts' не передан. Контекстно-зависимые метрики "
                "(faithfulness, context_precision) будут исключены из прогона."
            )
            # Отфильтровываем метрики, требующие контекст
            metrics_to_run = [
                m
                for m in self._metric_objects
                if m.name not in ["faithfulness", "context_precision"]
            ]

            if not metrics_to_run:
                logger.warning("RagasMetrics: после исключения контекстных метрик список пуст.")
                return {}

            contexts = [[""] for _ in prompts]

        data = {
            "question": prompts,
            "answer": generated,
            "ground_truth": targets,
            "contexts": contexts,
        }

        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate

            hf_dataset = Dataset.from_dict(data)
            result = ragas_evaluate(
                dataset=hf_dataset,
                metrics=metrics_to_run,
                raise_exceptions=self.raise_exceptions,
            )
            return {k: float(v) for k, v in dict(result).items()}

        except Exception as e:
            logger.error("RagasMetrics: сбой при вычислении — %s", e)
            return {}


# ------------------------------------------------------------------
# GenerationSpeedMetrics — латентность, throughput, token counts
# ------------------------------------------------------------------


class GenerationSpeedMetrics:
    """Бизнес-метрики скорости и стоимости инференса.

    Данные приходят не из текстов, а из ``extra["generation_stats"]`` —
    списка per-sample словарей, собранных в ``DecoderEvaluator._generate_chunks_with_stats``.

    Каждый элемент ``generation_stats`` содержит:
        - ``latency_s``       (float) — время генерации одного семпла, секунды
        - ``prompt_tokens``   (int)   — токены входного промпта
        - ``generated_tokens`` (int)  — сгенерированные токены (completion)

    Метрики:
        latency_mean_s, latency_p50_s, latency_p95_s, latency_p99_s
            — распределение латентности по семплам.
        throughput_tokens_per_s
            — суммарные сгенерированные токены / суммарное время;
              ключевая метрика для capacity planning.
        prompt_tokens_mean, generated_tokens_mean, total_tokens_mean
            — средние длины для мониторинга cost drift.
        empty_response_rate
            — доля пустых (после strip) ответов; >0.05 — тревожный сигнал.

    Args:
        warn_latency_p95_s: Если p95 латентности превышает порог — пишем WARNING.
            None отключает проверку.
        warn_empty_rate: Порог доли пустых ответов для WARNING. По умолчанию 0.05.
    """

    def __init__(
        self,
        warn_latency_p95_s: float | None = None,
        warn_empty_rate: float = 0.05,
    ) -> None:
        self.warn_latency_p95_s = warn_latency_p95_s
        self.warn_empty_rate = warn_empty_rate

    def compute(
        self,
        prompts: list[str],
        generated: list[str],
        targets: list[str],
        contexts: list[list[str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        stats = (extra or {}).get("generation_stats")
        if not stats:
            logger.warning(
                "GenerationSpeedMetrics: 'generation_stats' не найден в extra — "
                "метрики скорости недоступны. Убедитесь, что DecoderEvaluator "
                "использует _generate_chunks_with_stats."
            )
            return {}

        latencies = [s["latency_s"] for s in stats]
        prompt_tok = [s["prompt_tokens"] for s in stats]
        gen_tok = [s["generated_tokens"] for s in stats]

        sorted_lat = sorted(latencies)

        def _percentile(data: list[float], p: float) -> float:
            idx = max(0, int(len(data) * p / 100) - 1)
            return data[min(idx, len(data) - 1)]

        total_gen_tokens = sum(gen_tok)
        total_time = sum(latencies)
        throughput = total_gen_tokens / total_time if total_time > 0 else 0.0

        empty_count = sum(1 for g in generated if not g.strip())
        empty_rate = empty_count / len(generated) if generated else 0.0

        results = {
            "latency_mean_s": statistics.mean(latencies),
            "latency_p50_s": _percentile(sorted_lat, 50),
            "latency_p95_s": _percentile(sorted_lat, 95),
            "latency_p99_s": _percentile(sorted_lat, 99),
            "throughput_tokens_per_s": throughput,
            "prompt_tokens_mean": statistics.mean(prompt_tok) if prompt_tok else 0.0,
            "generated_tokens_mean": statistics.mean(gen_tok) if gen_tok else 0.0,
            "total_tokens_mean": statistics.mean(
                [p + g for p, g in zip(prompt_tok, gen_tok, strict=True)]
            )
            if prompt_tok
            else 0.0,
            "empty_response_rate": empty_rate,
        }

        p95 = results["latency_p95_s"]
        if self.warn_latency_p95_s is not None and p95 > self.warn_latency_p95_s:
            logger.warning(
                "GenerationSpeedMetrics: p95 латентность %.2f с превышает порог %.2f с.",
                p95,
                self.warn_latency_p95_s,
            )

        if empty_rate > self.warn_empty_rate:
            logger.warning(
                "GenerationSpeedMetrics: %.1f%% пустых ответов — превышен порог %.1f%%.",
                empty_rate * 100,
                self.warn_empty_rate * 100,
            )

        return results


# ------------------------------------------------------------------
# TextQualityMetrics — repetition, diversity, length ratio
# ------------------------------------------------------------------


class TextQualityMetrics:
    """Бизнес-метрики качества текста без LLM-судьи.

    Вычисляются только по сгенерированным текстам (и таргетам для length ratio).
    Быстрые, не требуют API. Подходят для каждой val-эпохи.

    Метрики:
        repetition_rate
            — средняя доля повторяющихся n-gram внутри каждого ответа.
              Высокое значение (>0.3) указывает на degeneration / mode collapse.
              Формула: 1 - unique_ngrams / total_ngrams per sample, усреднённое.
        corpus_unique_bigram_ratio
            — доля уникальных биграм по всему корпусу ответов.
              Мерит diversity на уровне батча: если модель генерирует одно и то же,
              это упадёт к 0.
        length_ratio_mean, length_ratio_std
            — отношение len(generated) / len(target) в словах, среднее и std.
              Значение ~1.0 — ideal, <0.5 — усечённые ответы, >2.0 — многословие.
        avg_gen_length_words
            — средняя длина ответа в словах (переехало сюда из DecoderEvaluator).
        avg_gen_length_chars
            — средняя длина в символах (удобно для cost-per-char мониторинга).

    Args:
        repetition_ngram_n: Размер n-gram для подсчёта repetition. По умолчанию 3.
    """

    def __init__(self, repetition_ngram_n: int = 3) -> None:
        self.n = repetition_ngram_n

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _repetition_rate(self, text: str) -> float:
        tokens = text.split()
        if len(tokens) < self.n:
            return 0.0
        grams = self._ngrams(tokens, self.n)
        if not grams:
            return 0.0
        return 1.0 - len(set(grams)) / len(grams)

    def compute(
        self,
        prompts: list[str],
        generated: list[str],
        targets: list[str],
        contexts: list[list[str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if not generated:
            return {}

        # --- Repetition rate (per-sample, затем mean) ---
        rep_rates = [self._repetition_rate(g) for g in generated]
        repetition_rate = statistics.mean(rep_rates)

        # --- Corpus-level bigram diversity ---
        all_bigrams: list[tuple[str, ...]] = []
        for g in generated:
            tokens = g.split()
            all_bigrams.extend(self._ngrams(tokens, 2))
        corpus_unique_bigram_ratio = (
            len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
        )

        # --- Length ratio (words) ---
        ratios: list[float] = []
        for g, t in zip(generated, targets, strict=True):
            target_len = len(t.split())
            if target_len > 0:
                ratios.append(len(g.split()) / target_len)

        length_ratio_mean = statistics.mean(ratios) if ratios else 0.0
        length_ratio_std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0

        # --- Avg lengths ---
        word_lengths = [len(g.split()) for g in generated]
        char_lengths = [len(g) for g in generated]

        return {
            "repetition_rate": repetition_rate,
            "corpus_unique_bigram_ratio": corpus_unique_bigram_ratio,
            "length_ratio_mean": length_ratio_mean,
            "length_ratio_std": length_ratio_std,
            "avg_gen_length_words": statistics.mean(word_lengths),
            "avg_gen_length_chars": float(statistics.mean(char_lengths)),
        }
