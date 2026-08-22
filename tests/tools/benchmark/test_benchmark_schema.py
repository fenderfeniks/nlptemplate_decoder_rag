import json

import pytest

from src.tools.benchmark.schema import BenchmarkDataset, BenchmarkRecord


class TestBenchmarkRecord:
    def test_to_and_from_dict(self):
        """Проверка сериализации и десериализации одной записи."""
        original_data = {
            "chunk_id": "abcdef1234567890",
            "chunk_text": "Sample context text.",
            "question": "What is sample?",
            "answer": "Context text.",
            "nli_score": 0.95,
            "metadata": {"source": "wiki"},
            "generator_model": "llama-3",
        }

        record = BenchmarkRecord.from_dict(original_data)

        assert record.chunk_id == "abcdef1234567890"
        assert record.nli_score == 0.95

        exported_data = record.to_dict()
        assert exported_data == original_data

    def test_default_fields(self):
        """Проверка дефолтных значений для опциональных полей."""
        record = BenchmarkRecord(
            chunk_id="123", chunk_text="text", question="Q", answer="A", nli_score=1.0
        )
        assert record.metadata == {}
        assert record.generator_model == ""


class TestBenchmarkDataset:
    @pytest.fixture
    def sample_dataset(self):
        ds = BenchmarkDataset()
        ds.append(BenchmarkRecord("id1", "txt1", "Q1", "A1", 0.9, generator_model="model1"))
        ds.append(BenchmarkRecord("id2", "txt2", "Q2", "A2", 0.7, generator_model="model2"))
        ds.append(BenchmarkRecord("id3", "txt3", "Q3", "A3", 0.8, generator_model="model1"))
        return ds

    def test_collection_magic_methods(self, sample_dataset):
        """Проверка len() и итерирования."""
        assert len(sample_dataset) == 3

        items = list(sample_dataset)
        assert len(items) == 3
        assert items[0].chunk_id == "id1"

    def test_summary_calculation(self, sample_dataset):
        """Проверка вычисления статистики: min, max, mean и сбор уникальных значений."""
        summary = sample_dataset.summary()

        assert summary["total"] == 3
        assert summary["unique_chunks"] == 3
        # Mean: (0.9 + 0.7 + 0.8) / 3 = 0.8
        assert pytest.approx(summary["nli_score_mean"]) == 0.8
        assert summary["nli_score_min"] == 0.7
        assert summary["nli_score_max"] == 0.9

        # Сбор уникальных моделей генерации
        assert sorted(summary["models_used"]) == ["model1", "model2"]

    def test_summary_empty(self):
        """Проверка summary для пустого датасета."""
        ds = BenchmarkDataset()
        assert ds.summary() == {"total": 0}

    def test_save_and_load_jsonl(self, sample_dataset, tmp_path):
        """Проверка полного цикла записи на диск и чтения (Roundtrip)."""
        file_path = tmp_path / "bench.jsonl"

        sample_dataset.save_jsonl(file_path)

        assert file_path.exists()

        loaded_ds = BenchmarkDataset.load_jsonl(file_path)
        assert len(loaded_ds) == 3
        assert loaded_ds.records[1].nli_score == 0.7
        assert loaded_ds.records[2].question == "Q3"

    def test_load_jsonl_skips_empty_lines(self, tmp_path):
        """Чтение должно игнорировать пустые строки в JSONL."""
        file_path = tmp_path / "dirty.jsonl"
        valid_json = json.dumps(
            {"chunk_id": "1", "chunk_text": "t", "question": "q", "answer": "a", "nli_score": 1.0}
        )

        with open(file_path, "w") as f:
            f.write(f"{valid_json}\n\n   \n{valid_json}\n")

        loaded_ds = BenchmarkDataset.load_jsonl(file_path)
        assert len(loaded_ds) == 2

    def test_load_jsonl_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            BenchmarkDataset.load_jsonl("non_existent_file.jsonl")
