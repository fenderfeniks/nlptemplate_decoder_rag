from unittest.mock import MagicMock

import pytest

# Укажи правильный путь импорта
from src.evaluation.metrics.generator import (
    GenerationSpeedMetrics,
    GeneratorMetricsPipeline,
    RagasMetrics,
    StatisticalMetrics,
    TextQualityMetrics,
)


# ===========================================================================
# Тесты Пайплайна
# ===========================================================================


class MockPassMetric:
    def compute(self, prompts, generated, targets, contexts=None, extra=None):
        return {"pass_score": 1.0}


class MockFailMetric:
    def compute(self, prompts, generated, targets, contexts=None, extra=None):
        raise ValueError("Fatal computation error")


class TestGeneratorMetricsPipeline:
    def test_compute_all_isolates_failures(self):
        """Пайплайн должен ловить исключения и возвращать результаты успешных метрик."""
        pipeline = GeneratorMetricsPipeline([MockFailMetric(), MockPassMetric()])

        results = pipeline.compute_all(prompts=["Q1"], generated=["A1"], targets=["T1"])

        assert "pass_score" in results
        assert results["pass_score"] == 1.0
        assert "fail_score" not in results


# ===========================================================================
# Тесты Статистических Метрик
# ===========================================================================


class TestStatisticalMetrics:
    def test_extract_score_formats(self):
        """Проверка парсинга разных ответов от HuggingFace evaluate."""

        # Старый формат
        class DummyScore:
            class Mid:
                fmeasure = 0.85

            mid = Mid()

        assert StatisticalMetrics._extract_score(DummyScore()) == 0.85
        assert StatisticalMetrics._extract_score([0.9]) == 0.9
        assert StatisticalMetrics._extract_score(0.7) == 0.7

    def test_compute_skips_unloaded_metrics(self, mocker):
        """Проверка ленивой инициализации и пропуска отключенных метрик."""
        # Мокаем hf_evaluate.load, чтобы он ничего не грузил
        mocker.patch("src.evaluation.metrics.generator.hf_evaluate.load", return_value=None)

        metrics = StatisticalMetrics(use_rouge=False, use_bleu=False)
        res = metrics.compute(["p"], ["g"], ["t"])
        assert res == {}


# ===========================================================================
# Тесты Ragas Метрик
# ===========================================================================


class TestRagasMetrics:
    @pytest.fixture
    def mock_ragas_deps(self, mocker):
        """Мокает тяжелые зависимости, импортируемые локально внутри методов RagasMetrics."""
        mock_ragas = MagicMock()
        mock_datasets = MagicMock()
        mock_importlib = mocker.patch("importlib.import_module")

        # Инжектим моки в sys.modules, чтобы локальные импорты внутри compute() подхватили их
        mocker.patch.dict("sys.modules", {"ragas": mock_ragas, "datasets": mock_datasets})
        return mock_importlib, mock_ragas, mock_datasets

    def test_context_dependent_metrics_filtered_when_no_context(self, mock_ragas_deps):
        """Если контекст не передан, faithfulness и context_precision должны отсекаться."""
        mock_importlib, mock_ragas, _ = mock_ragas_deps

        # Симулируем загруженные инстансы метрик
        metric_faith = MagicMock(name="faithfulness")
        metric_faith.name = "faithfulness"

        metric_answer = MagicMock(name="answer_correctness")
        metric_answer.name = "answer_correctness"

        metrics = RagasMetrics(["faithfulness", "answer_correctness"])
        # Руками заполняем загруженные объекты, чтобы обойти реальный import_module
        metrics._metric_objects = [metric_faith, metric_answer]
        metrics._initialized = True

        mock_ragas.evaluate.return_value = {"answer_correctness": 0.9}

        res = metrics.compute(prompts=["Q"], generated=["A"], targets=["T"], contexts=None)

        # Проверяем, что в evaluate ушла только answer_correctness
        called_metrics = mock_ragas.evaluate.call_args.kwargs["metrics"]
        assert len(called_metrics) == 1
        assert called_metrics[0].name == "answer_correctness"
        assert res == {"answer_correctness": 0.9}


# ===========================================================================
# Тесты Скорости Генерации (Бизнес-метрики)
# ===========================================================================


class TestGenerationSpeedMetrics:
    def test_compute_with_valid_stats(self):
        """Математическая проверка throughput, средних значений и перцентилей."""
        metrics = GenerationSpeedMetrics()

        extra = {
            "generation_stats": [
                {"latency_s": 1.0, "prompt_tokens": 10, "generated_tokens": 20},
                {"latency_s": 3.0, "prompt_tokens": 20, "generated_tokens": 60},
            ]
        }

        res = metrics.compute(
            prompts=["Q1", "Q2"], generated=["A1", "A2"], targets=["T1", "T2"], extra=extra
        )

        # Throughput: (20 + 60) / (1.0 + 3.0) = 80 / 4.0 = 20.0
        assert res["throughput_tokens_per_s"] == 20.0

        # Tokens mean
        assert res["prompt_tokens_mean"] == 15.0
        assert res["generated_tokens_mean"] == 40.0
        assert res["total_tokens_mean"] == 55.0  # (30 + 80) / 2

        # Latency (кастомный расчет перцентилей в коде)
        assert res["latency_mean_s"] == 2.0
        assert res["empty_response_rate"] == 0.0

    def test_empty_response_rate(self):
        """Проверка подсчета пустых ответов."""
        metrics = GenerationSpeedMetrics()

        extra = {
            "generation_stats": [
                {"latency_s": 1.0, "prompt_tokens": 10, "generated_tokens": 5},
                {"latency_s": 1.0, "prompt_tokens": 10, "generated_tokens": 0},
            ]
        }

        # Один пустой, один нормальный
        res = metrics.compute(
            prompts=["Q1", "Q2"], generated=["Valid", "   "], targets=["T1", "T2"], extra=extra
        )

        assert res["empty_response_rate"] == 0.5

    def test_missing_extra_returns_empty(self):
        """Если generation_stats не передан, метрика возвращает {}."""
        metrics = GenerationSpeedMetrics()
        assert metrics.compute(["p"], ["g"], ["t"]) == {}


# ===========================================================================
# Тесты Качества Текста (Бизнес-метрики)
# ===========================================================================


class TestTextQualityMetrics:
    def test_repetition_rate(self):
        """Проверка выявления дегенерации n-gram."""
        metrics = TextQualityMetrics(repetition_ngram_n=3)

        # Нет повторений (слишком короткий текст)
        assert metrics._repetition_rate("a b") == 0.0

        # Нет повторений
        assert metrics._repetition_rate("this is a unique sentence") == 0.0

        # Одно повторение: "a a a" (2 триграммы, 1 уникальная)
        # 1.0 - (1 / 2) = 0.5
        assert metrics._repetition_rate("a a a a") == 0.5

    def test_compute_quality_metrics(self):
        """Комплексная проверка ratio и diversity."""
        metrics = TextQualityMetrics()

        generated = ["hello world", "just a test"]
        targets = ["hi world", "a longer test response"]

        res = metrics.compute(generated=generated, prompts=[], targets=targets)

        # Average lengths (2 words in first, 3 in second) -> 2.5
        assert res["avg_gen_length_words"] == 2.5

        # Length ratios: 2/2=1.0 and 3/4=0.75 -> mean = 0.875
        assert res["length_ratio_mean"] == 0.875

        # Bigrams: ("hello", "world"), ("just", "a"), ("a", "test")
        # Total bigrams = 3. Unique bigrams = 3. Ratio = 1.0
        assert res["corpus_unique_bigram_ratio"] == 1.0
