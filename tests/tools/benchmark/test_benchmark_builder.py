from unittest.mock import MagicMock

import pytest

# Укажи правильный путь импорта
from src.tools.benchmark.builder import BenchmarkBuilder
from src.tools.benchmark.schema import BenchmarkRecord
from src.tools.evaluation.schema import EvalResult


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    # Дефолтное успешное поведение
    gen.generate.return_value = ("Сгенерированный вопрос?", "Ответ")
    return gen


@pytest.fixture
def mock_nli_judge():
    judge = MagicMock()
    # Дефолтное успешное поведение (score >= threshold)
    judge.evaluate.return_value = EvalResult(score=0.85, verdict=True)
    return judge


@pytest.fixture
def builder(mock_generator, mock_nli_judge):
    return BenchmarkBuilder(
        generator=mock_generator,
        nli_judge=mock_nli_judge,
        nli_threshold=0.60,
        max_samples_per_chunk=1,
        min_chunk_length=10,  # Уменьшено для удобства тестирования
    )


# ===========================================================================
# Тесты утилит и фильтров
# ===========================================================================


class TestBenchmarkBuilderUtils:
    def test_compute_chunk_id(self):
        """Проверка детерминированности алгоритма SHA-256[:16]."""
        text = "Hello world"
        meta1 = {"url": "http://example.com", "title": "Test"}

        chunk_id1 = BenchmarkBuilder.compute_chunk_id(text, meta1)
        chunk_id2 = BenchmarkBuilder.compute_chunk_id(text, meta1)

        assert len(chunk_id1) == 16
        assert chunk_id1 == chunk_id2

        # При отсутствии метаданных хэш должен считаться корректно (с пустыми строками)
        chunk_id_empty_meta = BenchmarkBuilder.compute_chunk_id(text, {})
        assert chunk_id_empty_meta != chunk_id1
        assert len(chunk_id_empty_meta) == 16

    def test_is_chunk_too_short(self, builder):
        assert builder._is_chunk_too_short("short") is True
        assert builder._is_chunk_too_short("This is a long enough chunk") is False

    def test_nli_filter_pass_and_fail(self, builder, mock_nli_judge):
        # 1. Проходит порог
        mock_nli_judge.evaluate.return_value = EvalResult(score=0.7)
        passed, score = builder._nli_filter("chunk", "answer", "id1")
        assert passed is True
        assert score == 0.7

        # 2. Не проходит порог
        mock_nli_judge.evaluate.return_value = EvalResult(score=0.5)
        passed, score = builder._nli_filter("chunk", "answer", "id1")
        assert passed is False
        assert score == 0.5

        # 3. NLI вернул None (ошибка модели)
        mock_nli_judge.evaluate.return_value = EvalResult(score=None)
        passed, score = builder._nli_filter("chunk", "answer", "id1")
        assert passed is False
        assert score == 0.0


# ===========================================================================
# Тесты ядра логики обработки (_process_chunk)
# ===========================================================================


class TestBenchmarkBuilderProcessing:
    def test_process_chunk_skipped_short(self, builder):
        stats = {"skipped_short": 0}
        records = builder._process_chunk("short", {}, "id1", stats, "model_A")

        assert len(records) == 0
        assert stats["skipped_short"] == 1

    def test_process_chunk_failed_generation(self, builder, mock_generator):
        mock_generator.generate.return_value = None
        stats = {"failed_generation": 0}

        records = builder._process_chunk("Valid long chunk text", {}, "id1", stats, "model_A")

        assert len(records) == 0
        assert stats["failed_generation"] == 1

    def test_process_chunk_failed_nli(self, builder, mock_nli_judge):
        mock_nli_judge.evaluate.return_value = EvalResult(score=0.1)  # Меньше 0.6
        stats = {"generated": 0, "failed_nli": 0}

        records = builder._process_chunk("Valid long chunk text", {}, "id1", stats, "model_A")

        assert len(records) == 0
        assert stats["generated"] == 1
        assert stats["failed_nli"] == 1

    def test_process_chunk_success_multiple_samples(self, builder):
        builder.max_samples_per_chunk = 2
        stats = {"generated": 0, "accepted": 0}

        records = builder._process_chunk(
            "Valid long chunk text", {"url": "x"}, "id1", stats, "model_A"
        )

        assert len(records) == 2
        assert stats["generated"] == 2
        assert stats["accepted"] == 2

        # Проверяем структуру возвращаемого объекта
        assert isinstance(records[0], BenchmarkRecord)
        assert records[0].chunk_id == "id1"
        assert records[0].generator_model == "model_A"
        assert records[0].metadata == {"url": "x"}


# ===========================================================================
# Тесты внешних интерфейсов с итераторами
# ===========================================================================


class TestBenchmarkBuilderIterators:
    def test_build_from_dataset(self, builder):
        """Тест работы с объектом, имитирующим HuggingFace Dataset."""
        # Имитируем датасет с колонками text и chunk_id
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text", "chunk_id", "meta_field"]
        mock_dataset.__iter__.return_value = [
            {"text": "A valid chunk text here.", "chunk_id": "c1", "meta_field": "val1"},
            {
                "text": "tiny",
                "chunk_id": "c2",
                "meta_field": "val2",
            },  # Пропустится (слишком короткий)
            {"text": "Another valid chunk right here.", "chunk_id": "c3", "meta_field": "val3"},
        ]

        dataset_obj = builder.build_from_dataset(
            dataset=mock_dataset,
            text_column="text",
            id_column="chunk_id",
            generator_model_name="gen_model",
        )

        assert len(dataset_obj) == 2

        # Проверяем, что служебные колонки удалены из metadata
        assert dataset_obj.records[0].chunk_id == "c1"
        assert dataset_obj.records[0].metadata == {"meta_field": "val1"}
        assert dataset_obj.records[1].chunk_id == "c3"

    def test_build_from_dataset_no_id_column(self, builder):
        """Если id_column нет, chunk_id должен вычисляться на лету."""
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__iter__.return_value = [{"text": "Valid chunk without precomputed id."}]

        dataset_obj = builder.build_from_dataset(mock_dataset)

        assert len(dataset_obj) == 1
        # ID должен быть вычислен через compute_chunk_id
        expected_id = builder.compute_chunk_id("Valid chunk without precomputed id.", {})
        assert dataset_obj.records[0].chunk_id == expected_id

    def test_build_from_dataloader(self, builder):
        """Тест устаревшего метода работы через PyTorch DataLoader."""
        # Имитируем батчи из DataLoader
        batch1 = {
            "input_ids": [1, 2],  # batch_len = 2
            "text": ["First valid text in batch.", "Second valid text in batch."],
            "metadata": [{"url": "1"}, {"url": "2"}],
        }
        mock_dataloader = [batch1]

        dataset_obj = builder.build_from_dataloader(mock_dataloader)

        assert len(dataset_obj) == 2
        assert dataset_obj.records[0].chunk_text == "First valid text in batch."
        assert dataset_obj.records[0].metadata == {"url": "1"}
