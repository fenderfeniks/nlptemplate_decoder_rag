import numpy as np
import pytest

# Укажи правильный путь импорта
from src.evaluation.metrics.retriever import RetrieverMetrics


# ===========================================================================
# Тесты вспомогательных математических функций
# ===========================================================================


class TestRetrieverMathUtils:
    @pytest.fixture
    def metrics(self):
        return RetrieverMetrics(retrieval_top_k=5, rerank_top_k=3)

    def test_jaccard_similarity(self, metrics):
        """Проверка расчета индекса Жаккара."""
        # Полное совпадение
        assert metrics._jaccard_similarity("hello world", "World Hello") == 1.0
        # Полное несовпадение
        assert metrics._jaccard_similarity("hello", "world") == 0.0
        # Частичное совпадение (intersection=1, union=3 -> 1/3)
        assert pytest.approx(metrics._jaccard_similarity("cat dog", "dog mouse")) == 0.3333333
        # Пустые строки
        assert metrics._jaccard_similarity("", "world") == 0.0
        assert metrics._jaccard_similarity("", "") == 0.0

    def test_compute_redundancy(self, metrics):
        """Проверка среднего попарного сходства (Redundancy)."""
        results = [
            {"metadata": {"text": "A B"}},
            {"metadata": {"text": "B C"}},
            {"metadata": {"text": "A C"}},
        ]
        # Пары: (A B, B C)->1/3; (A B, A C)->1/3; (B C, A C)->1/3. Среднее = 1/3
        redundancy = metrics._compute_redundancy(results)
        assert pytest.approx(redundancy) == 0.3333333

        # Меньше двух результатов — избыточность 0
        assert metrics._compute_redundancy([{"metadata": {"text": "A B"}}]) == 0.0


# ===========================================================================
# Тесты расчета Information Retrieval (IR) метрик
# ===========================================================================


class TestIRMetrics:
    @pytest.fixture
    def metrics(self):
        return RetrieverMetrics(retrieval_top_k=10)

    def test_ir_metrics_calculation(self, metrics):
        """
        Проверка точного математического расчета MRR, Precision, NDCG.

        Сценарий:
        - Истинные релевантные ID: doc1, doc2, doc3
        - Выдача: [doc4, doc1, doc5, doc2]

        Ожидания:
        - Hits = 2 (doc1, doc2)
        - MRR = первый хит на 2 позиции (индекс 1) -> 1/2 = 0.5
        - Precision = 2 / 4 = 0.5
        - DCG = 1/log2(1+2) + 1/log2(3+2) = 1/1.58496 + 1/2.32192 ≈ 0.6309 + 0.4306 ≈ 1.0615
        - IDCG (для 3 релевантных, но длина выдачи 4, значит считаем 3 слагаемых):
          1/log2(0+2) + 1/log2(1+2) + 1/log2(2+2) = 1 + 0.6309 + 0.5 = 2.1309
        - NDCG = 1.0615 / 2.1309 ≈ 0.4981
        """
        results = [
            {"metadata": {"doc_id": "doc4"}},
            {"metadata": {"doc_id": "doc1"}},
            {"metadata": {"doc_id": "doc5"}},
            {"metadata": {"doc_id": "doc2"}},
        ]
        true_ids = {"doc1", "doc2", "doc3"}

        mrr, prec, ndcg, hits = metrics._ir_metrics(results, true_ids, "score")

        assert hits == 2
        assert mrr == 0.5
        assert prec == 0.5

        expected_dcg = (1.0 / np.log2(3)) + (1.0 / np.log2(5))
        expected_idcg = 1.0 + (1.0 / np.log2(3)) + (1.0 / np.log2(4))
        expected_ndcg = expected_dcg / expected_idcg

        assert pytest.approx(ndcg) == expected_ndcg

    def test_ir_metrics_empty(self, metrics):
        """Проверка расчета, если выдача пустая или релевантных доков нет."""
        mrr, prec, ndcg, hits = metrics._ir_metrics([], {"doc1"}, "score")
        assert (mrr, prec, ndcg, hits) == (0.0, 0.0, 0.0, 0)

        mrr, prec, ndcg, hits = metrics._ir_metrics(
            [{"metadata": {"doc_id": "doc2"}}], set(), "score"
        )
        assert (mrr, prec, ndcg, hits) == (0.0, 0.0, 0.0, 0)


# ===========================================================================
# Тесты полного пайплайна метрик (compute)
# ===========================================================================


class TestRetrieverComputePipeline:
    def test_empty_results_and_errors(self):
        metrics = RetrieverMetrics(retrieval_top_k=5)

        # 0 запросов
        res = metrics.compute([], [])
        assert res["recall_5_biencoder"] == 0.0
        assert res["empty_retrieval_rate"] == 1.0

        # Несовпадение длин
        with pytest.raises(ValueError, match="не совпадает с ground_truth_ids"):
            metrics.compute([[]], [[1], [2]])

    def test_compute_with_reranker(self):
        """Проверка пайплайна: bi-encoder recall считается по top_K, а остальные - по rerank_K."""
        metrics = RetrieverMetrics(retrieval_top_k=3, rerank_top_k=2)

        # Один запрос. Истинный док: "target"
        ground_truth = [["target"]]

        # Ретривер нашел его на 3-м месте (после реранка он стал на 1-е)
        search_results = [
            [
                {"score": 0.8, "cross_encoder_score": 0.9, "metadata": {"doc_id": "target"}},
                {"score": 0.9, "cross_encoder_score": 0.5, "metadata": {"doc_id": "noise1"}},
                {"score": 0.7, "cross_encoder_score": 0.2, "metadata": {"doc_id": "noise2"}},
            ]
        ]

        res = metrics.compute(search_results, ground_truth)

        # Bi-encoder Recall@3 должен быть 1.0 (target есть в топ-3)
        assert res["recall_3_biencoder"] == 1.0
        # Final Recall@2 должен быть 1.0 (target есть в топ-2)
        assert res["recall_2_final"] == 1.0
        # MRR = 1.0, т.к. после реранкинга (по порядку в списке) target на 1 месте
        assert res["mrr"] == 1.0
        assert res["precision_2"] == 0.5  # 1 хит из 2 доков в срезе rerank_top_k

    def test_ood_and_similarity_thresholds(self):
        """Проверка работы порогов фильтрации и OOD."""
        metrics = RetrieverMetrics(retrieval_top_k=2, similarity_threshold=0.5, ood_threshold=0.8)

        ground_truth = [["target1"], ["target2"]]

        # Запрос 1: OOD (top score 0.7 < 0.8), но проходит similarity_threshold
        # Запрос 2: Не проходит similarity_threshold (score 0.4 < 0.5) -> пустая выдача
        search_results = [
            [{"score": 0.7, "metadata": {"doc_id": "target1"}}],
            [{"score": 0.4, "metadata": {"doc_id": "target2"}}],
        ]

        res = metrics.compute(search_results, ground_truth)

        # OOD rate: 1 из 2 (Запрос 1)
        assert res["ood_rate"] == 0.5
        # Empty rate: 1 из 2 (Запрос 2 после фильтрации)
        assert res["empty_retrieval_rate"] == 0.5
        # Recall: Запрос 1 нашел таргет, Запрос 2 пуст -> 0.5
        assert res["recall_2_biencoder"] == 0.5
