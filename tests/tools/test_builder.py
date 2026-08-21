# tests/tools/benchmark/test_builder.py
"""Тесты для BenchmarkBuilder, BenchmarkDataset и compute_chunk_id."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.benchmark.builder import BenchmarkBuilder
from src.tools.benchmark.schema import BenchmarkDataset, BenchmarkRecord


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def mock_generator():
    """Генератор, который всегда возвращает фиксированную QA-пару."""
    gen = MagicMock()
    gen.generate.return_value = ("What is the capital of France?", "Paris")
    return gen


@pytest.fixture()
def mock_nli_judge_pass():
    """NLI-judge, который всегда одобряет (score=0.9)."""
    from src.tools.evaluation.schema import EvalResult
    judge = MagicMock()
    judge.evaluate.return_value = EvalResult(score=0.9, verdict=True)
    return judge


@pytest.fixture()
def mock_nli_judge_fail():
    """NLI-judge, который всегда отклоняет (score=0.1)."""
    from src.tools.evaluation.schema import EvalResult
    judge = MagicMock()
    judge.evaluate.return_value = EvalResult(score=0.1, verdict=False)
    return judge


def _make_dataloader(texts: list[str], metadatas: list[dict] | None = None):
    """Создаёт минимальный mock DataLoader с полями text и metadata."""
    if metadatas is None:
        metadatas = [{} for _ in texts]

    batch = {
        "input_ids": [[1]] * len(texts),
        "text": texts,
        "metadata": metadatas,
    }
    return [batch]  # один батч


# ------------------------------------------------------------------
# compute_chunk_id
# ------------------------------------------------------------------

class TestComputeChunkId:
    def test_matches_indexer_algorithm(self):
        """chunk_id должен совпадать с KnowledgeBaseIndexer._generate_doc_id."""
        text = "Paris is the capital of France."
        meta = {"url": "https://example.com", "title": "France"}
        composite = f"{text}_{meta.get('url', '')}_{meta.get('title', '')}"
        expected = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]
        assert BenchmarkBuilder.compute_chunk_id(text, meta) == expected

    def test_empty_metadata(self):
        """Пустые метаданные не вызывают ошибку."""
        chunk_id = BenchmarkBuilder.compute_chunk_id("Some text", {})
        assert len(chunk_id) == 16

    def test_deterministic(self):
        """Тот же вход — тот же chunk_id."""
        text, meta = "hello world", {"url": "x"}
        assert BenchmarkBuilder.compute_chunk_id(text, meta) == \
               BenchmarkBuilder.compute_chunk_id(text, meta)

    def test_different_texts_different_ids(self):
        """Разные тексты — разные chunk_id."""
        assert BenchmarkBuilder.compute_chunk_id("text A", {}) != \
               BenchmarkBuilder.compute_chunk_id("text B", {})


# ------------------------------------------------------------------
# BenchmarkBuilder
# ------------------------------------------------------------------

class TestBenchmarkBuilder:
    def _make_builder(self, generator, nli_judge, nli_threshold=0.60):
        return BenchmarkBuilder(
            generator=generator,
            nli_judge=nli_judge,
            nli_threshold=nli_threshold,
            min_chunk_length=10,
        )

    def test_accepted_record_has_correct_fields(self, mock_generator, mock_nli_judge_pass):
        builder = self._make_builder(mock_generator, mock_nli_judge_pass)
        texts = ["Paris is the capital of France and a major European city."]
        dataloader = _make_dataloader(texts, [{"url": "http://test.com", "title": "France"}])

        dataset = builder.build_from_dataloader(dataloader)

        assert len(dataset) == 1
        record = dataset.records[0]
        assert record.question == "What is the capital of France?"
        assert record.answer == "Paris"
        assert record.nli_score == pytest.approx(0.9)
        assert len(record.chunk_id) == 16

    def test_nli_filter_rejects_below_threshold(self, mock_generator, mock_nli_judge_fail):
        builder = self._make_builder(mock_generator, mock_nli_judge_fail, nli_threshold=0.60)
        texts = ["Some sufficiently long text about something interesting."]
        dataloader = _make_dataloader(texts)

        dataset = builder.build_from_dataloader(dataloader)

        assert len(dataset) == 0

    def test_short_chunks_skipped(self, mock_generator, mock_nli_judge_pass):
        builder = BenchmarkBuilder(
            generator=mock_generator,
            nli_judge=mock_nli_judge_pass,
            min_chunk_length=100,
        )
        texts = ["short"]  # 5 символов < 100
        dataloader = _make_dataloader(texts)

        dataset = builder.build_from_dataloader(dataloader)

        assert len(dataset) == 0
        mock_generator.generate.assert_not_called()

    def test_generator_failure_handled_gracefully(self, mock_nli_judge_pass):
        """Если генератор вернул None — запись пропускается, не падает."""
        gen = MagicMock()
        gen.generate.return_value = None

        builder = self._make_builder(gen, mock_nli_judge_pass)
        texts = ["Some sufficiently long text about something interesting here."]
        dataloader = _make_dataloader(texts)

        dataset = builder.build_from_dataloader(dataloader)
        assert len(dataset) == 0

    def test_multiple_chunks(self, mock_generator, mock_nli_judge_pass):
        texts = [
            "The Eiffel Tower is located in Paris, France and is 330 meters tall.",
            "Python is a high-level programming language created by Guido van Rossum.",
        ]
        dataloader = _make_dataloader(texts)
        builder = self._make_builder(mock_generator, mock_nli_judge_pass)

        dataset = builder.build_from_dataloader(dataloader)
        assert len(dataset) == 2

    def test_chunk_id_consistent_with_metadata(self, mock_generator, mock_nli_judge_pass):
        """chunk_id в записи должен совпадать с compute_chunk_id(text, meta)."""
        text = "The Louvre Museum is the world's largest art museum in Paris."
        meta = {"url": "https://louvre.fr", "title": "Louvre"}
        dataloader = _make_dataloader([text], [meta])

        builder = self._make_builder(mock_generator, mock_nli_judge_pass)
        dataset = builder.build_from_dataloader(dataloader)

        assert len(dataset) == 1
        expected_id = BenchmarkBuilder.compute_chunk_id(text, meta)
        assert dataset.records[0].chunk_id == expected_id


# ------------------------------------------------------------------
# BenchmarkDataset (save/load)
# ------------------------------------------------------------------

class TestBenchmarkDataset:
    def _make_record(self, chunk_id="abc123", question="Q?", answer="A.") -> BenchmarkRecord:
        return BenchmarkRecord(
            chunk_id=chunk_id,
            chunk_text="Some text about the topic.",
            question=question,
            answer=answer,
            nli_score=0.85,
            metadata={"url": "http://example.com"},
            generator_model="gpt-4o-mini",
        )

    def test_save_and_load_roundtrip(self):
        ds = BenchmarkDataset()
        ds.append(self._make_record("id1", "Q1?", "A1."))
        ds.append(self._make_record("id2", "Q2?", "A2."))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            ds.save_jsonl(path)

            loaded = BenchmarkDataset.load_jsonl(path)

        assert len(loaded) == 2
        assert loaded.records[0].chunk_id == "id1"
        assert loaded.records[1].answer == "A2."

    def test_jsonl_format(self):
        """Каждая строка JSONL должна быть валидным JSON."""
        ds = BenchmarkDataset()
        ds.append(self._make_record())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            ds.save_jsonl(path)
            lines = path.read_text(encoding="utf-8").strip().split("\n")

        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "chunk_id" in parsed
        assert "question" in parsed
        assert "nli_score" in parsed

    def test_summary(self):
        ds = BenchmarkDataset()
        ds.append(self._make_record("id1"))
        ds.append(self._make_record("id2"))
        summary = ds.summary()
        assert summary["total"] == 2
        assert "nli_score_mean" in summary

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            BenchmarkDataset.load_jsonl("/nonexistent/path/bench.jsonl")