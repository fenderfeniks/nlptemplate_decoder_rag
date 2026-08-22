from unittest.mock import MagicMock

import pytest

# Укажи правильный путь импорта
from src.evaluation.evaluators.retriever import RetrieverEvaluator


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    # Возвращаем фейковые результаты поиска для двух запросов
    retriever.batch_search.return_value = [
        [{"metadata": {"text": "Doc1"}}],
        [{"metadata": {"text": "Doc2"}}],
    ]
    return retriever


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    # Токенизатор будет "возвращать" список из 3 токенов на любой текст
    tokenizer.encode.return_value = [101, 202, 303]
    return tokenizer


@pytest.fixture
def mock_metrics_calculator(mocker):
    # Мокаем внутри самого модуля, чтобы Evaluator использовал наш мок
    mock_cls = mocker.patch("src.evaluation.evaluators.retriever.RetrieverMetrics")
    instance = mock_cls.return_value
    # Дефолтные метрики
    instance.compute.return_value = {"recall_20_biencoder": 0.8, "ndcg_5": 0.7}
    return instance


@pytest.fixture
def mock_experiment_logger():
    return MagicMock()


@pytest.fixture
def mock_sys_exit(mocker):
    return mocker.patch("src.evaluation.evaluators.retriever.sys.exit")


# ===========================================================================
# Тесты проверки SLA и бизнес-правил (Drift Check)
# ===========================================================================


class TestRetrieverEvaluatorDrift:
    def test_check_drift_passes(self):
        """Если все метрики укладываются в SLA, метод отрабатывает без ошибок."""
        evaluator = RetrieverEvaluator(
            retriever=MagicMock(),
            tokenizer=MagicMock(),
            drift_cfg={
                "max_latency_ms": 100,
                "max_context_tokens": 500,
                "min_ndcg": 0.5,
                "min_recall_biencoder": 0.5,
            },
        )

        metrics = {
            "total_latency_ms_per_query": 50,
            "avg_context_tokens": 250,
            "ndcg_5": 0.9,
            "recall_20_biencoder": 0.9,
        }

        # Должен пройти молча
        evaluator._check_drift(metrics)

    def test_check_drift_latency_fails_sys_exit(self, mock_sys_exit):
        """При raise_on_drift=False скрипт падает через sys.exit(1)."""
        evaluator = RetrieverEvaluator(
            retriever=MagicMock(),
            tokenizer=MagicMock(),
            drift_cfg={"max_latency_ms": 100},
            raise_on_drift=False,
        )

        metrics = {"total_latency_ms_per_query": 150}  # Превышение
        evaluator._check_drift(metrics)

        mock_sys_exit.assert_called_once_with(1)

    def test_check_drift_quality_fails_runtime_error(self):
        """При raise_on_drift=True скрипт кидает RuntimeError (используется при обучении)."""
        evaluator = RetrieverEvaluator(
            retriever=MagicMock(),
            tokenizer=MagicMock(),
            drift_cfg={"min_ndcg": 0.8},
            raise_on_drift=True,
            rerank_top_k=5,
        )

        metrics = {"ndcg_5": 0.6}  # Ниже порога

        with pytest.raises(RuntimeError, match=r"ДРИФТ \(Качество\)"):
            evaluator._check_drift(metrics)

    def test_check_drift_multiple_failures(self):
        """Проверка агрегации сразу нескольких нарушений."""
        evaluator = RetrieverEvaluator(
            retriever=MagicMock(),
            tokenizer=MagicMock(),
            drift_cfg={"max_context_tokens": 100, "min_recall_biencoder": 0.9},
            raise_on_drift=True,
            retrieval_top_k=10,
        )

        metrics = {
            "avg_context_tokens": 200,  # Провал бюджета
            "recall_10_biencoder": 0.5,  # Провал качества
        }

        with pytest.raises(RuntimeError) as exc_info:
            evaluator._check_drift(metrics)

        err_msg = str(exc_info.value)
        assert "ДРИФТ (Бюджет)" in err_msg
        assert "ДРИФТ (Bi-encoder)" in err_msg


# ===========================================================================
# Тесты основного пайплайна оценки
# ===========================================================================


class TestRetrieverEvaluatorEvaluate:
    def test_evaluate_empty_queries(self, mock_retriever, mock_tokenizer, mock_experiment_logger):
        """Пустой список запросов обрабатывается корректно (early return)."""
        evaluator = RetrieverEvaluator(mock_retriever, mock_tokenizer)

        result = evaluator.evaluate([], [], mock_experiment_logger)

        assert result == {}
        mock_retriever.batch_search.assert_not_called()

    def test_evaluate_full_pipeline(
        self,
        mocker,
        mock_retriever,
        mock_tokenizer,
        mock_metrics_calculator,
        mock_experiment_logger,
    ):
        """Комплексная проверка: поиск -> тайминги -> метрики -> токены -> логгер."""
        # Мокаем время, чтобы total_latency_sec составил ровно 2.0 секунды
        mocker.patch("time.perf_counter", side_effect=[0.0, 2.0])

        # Настраиваем компонентные тайминги внутри мока ретривера
        mock_retriever.last_batch_timing_sec = {"encode": 0.5, "search": 1.5}

        evaluator = RetrieverEvaluator(
            retriever=mock_retriever, tokenizer=mock_tokenizer, retrieval_top_k=20, rerank_top_k=5
        )

        queries = ["q1", "q2"]
        ground_truths = [["doc1"], ["doc2"]]

        metrics = evaluator.evaluate(queries, ground_truths, mock_experiment_logger)

        # 1. Проверяем вызов batch_search
        mock_retriever.batch_search.assert_called_once_with(queries=queries, top_k=20)

        # 2. Проверяем вызов compute
        mock_metrics_calculator.compute.assert_called_once_with(
            mock_retriever.batch_search.return_value, ground_truths
        )

        # 3. Проверяем подсчет токенов
        # 2 документа, каждый по 3 токена -> итого 6 токенов / 2 запроса = 3.0 avg
        assert mock_tokenizer.encode.call_count == 2
        assert metrics["avg_context_tokens"] == 3.0

        # 4. Проверяем тайминги (2 запроса, 2.0 секунды = 1.0 сек/запрос = 1000 мс)
        assert metrics["total_latency_ms_per_query"] == 1000.0
        assert metrics["encode_latency_ms_per_query"] == 250.0  # 0.5s / 2 * 1000
        assert metrics["search_latency_ms_per_query"] == 750.0  # 1.5s / 2 * 1000
        assert "rerank_latency_ms_per_query" not in metrics

        # 5. Проверяем логирование
        mock_experiment_logger.log_metrics.assert_called_once_with(metrics, stage="test", step=0)
