# tests/decoder_pipeline/core/data/test_deduplication.py
from datasets import Dataset

from src.decoder_pipeline.core.data.transforms.deduplication import (
    ExactDeduplicationTransform,
    MinHashDeduplicationTransform,
)


def _make_dataset(records: list[dict]) -> Dataset:
    keys = records[0].keys()
    return Dataset.from_dict({k: [r[k] for r in records] for k in keys})


class TestExactDeduplicationTransform:
    def test_removes_exact_duplicates(self):
        ds = _make_dataset([
            {"text": "Уникальный текст один"},
            {"text": "Дублирующийся текст"},
            {"text": "Дублирующийся текст"},
        ])
        transform = ExactDeduplicationTransform(text_column="text", num_proc=1)
        result = transform(ds)

        assert len(result) == 2
        assert result["text"] == ["Уникальный текст один", "Дублирующийся текст"]


class TestMinHashDeduplicationTransform:
    def test_removes_near_duplicates(self):
        ds = _make_dataset([
            {"text": "Машинное обучение это раздел искусственного интеллекта."},
            {"text": "Машинное обучение — это раздел искусственного интеллекта!"},  # Нечеткий дубликат
            {"text": "Совершенно другой текст про кулинарию и рецепты пирогов."},
        ])
        # Низкий пороговый threshold для демонстрации отлова похожих строк
        transform = MinHashDeduplicationTransform(text_column="text", threshold=0.7, num_proc=1)
        result = transform(ds)

        assert len(result) == 2