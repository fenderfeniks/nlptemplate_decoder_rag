# tests/pipelines/base/core/data/transforms/test_validation.py
import pytest
from typing import Any
from datasets import Dataset

from src.pipelines.base.core.data.transforms.validation import (
    BaseValidationTransform,
    CleaningTransform,
)

# 1. Вспомогательный класс для тестирования абстрактного BaseValidationTransform
class DummyValidationTransform(BaseValidationTransform):
    def _validate_mode(self) -> None:
        if self.mode != "test":
            raise ValueError("Неверный режим")

    def _get_required_columns(self) -> list[str]:
        return ["text", "metadata"]

    def _get_filter_column(self) -> str:
        return "is_valid"

    def _validate_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Фиктивная логика: валидны только записи с metadata > 1
        batch["is_valid"] = [m > 1 for m in batch["metadata"]]
        return batch

# 2. Вспомогательный класс для замены MagicMock (решает проблему с PicklingError)
class DummyPipeline:
    def __call__(self, text: str) -> str:
        return str(text).upper()


class TestBaseValidationTransform:
    def test_init_invalid_mode(self):
        """Проверка, что абстрактный метод _validate_mode вызывается при инициализации."""
        with pytest.raises(ValueError, match="Неверный режим"):
            DummyValidationTransform(mode="wrong_mode")

    def test_validation_applies_and_filters_correctly(self, sample_text_dataset: Dataset):
        """Проверка полного цикла map + filter на тестовой реализации."""
        transform = DummyValidationTransform(mode="test", num_proc=1, batch_size=2)
        result = transform(sample_text_dataset)
        
        # sample_text_dataset имеет metadata = [1, 2, 3, 4]
        # Должны остаться только 2, 3, 4
        assert len(result) == 3
        assert result["metadata"] == [2, 3, 4]

    def test_missing_required_columns_skipped(self, sample_text_dataset: Dataset):
        """Если нет обязательных колонок, датасет возвращается без изменений."""
        ds_without_metadata = sample_text_dataset.remove_columns("metadata")
        transform = DummyValidationTransform(mode="test")
        result = transform(ds_without_metadata)
        
        assert len(result) == len(ds_without_metadata)


class TestCleaningTransform:
    def test_init_removes_none_columns(self):
        """Проверка, что None от Hydra корректно удаляются при инициализации."""
        transform = CleaningTransform(
            pipeline=DummyPipeline(),
            columns_to_clean=["text", None, "prompt"]
        )
        assert transform.columns_to_clean == ["text", "prompt"]

    def test_cleaning_applies_correctly(self, sample_text_dataset: Dataset):
        """Проверка применения клинера (теперь без PicklingError)."""
        transform = CleaningTransform(
            pipeline=DummyPipeline(),
            columns_to_clean=["text"],
            num_proc=1
        )
        result = transform(sample_text_dataset)
        
        assert result["text"][0] == "ПРИМЕР ТЕКСТА 1"
        assert result["prompt"][0] == "p1"

    def test_missing_columns_skipped(self, sample_text_dataset: Dataset):
        """Проверка безопасного пропуска, если колонок нет в датасете."""
        transform = CleaningTransform(
            pipeline=DummyPipeline(),
            columns_to_clean=["missing_text_col"]
        )
        result = transform(sample_text_dataset)
        
        assert len(result) == len(sample_text_dataset)