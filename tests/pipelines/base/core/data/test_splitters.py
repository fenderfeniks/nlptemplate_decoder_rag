# tests/pipelines/base/core/data/test_splitters.py
import pytest
from datasets import Dataset, DatasetDict

from src.pipelines.base.core.data.splitters import RandomDatasetSplitter


@pytest.fixture
def dummy_dataset():
    """Фикстура датасета из 100 элементов для удобного подсчета долей."""
    return Dataset.from_dict({"text": [f"Текст {i}" for i in range(100)]})


class TestRandomDatasetSplitter:
    def test_invalid_negative_sizes(self):
        """Проверка исключения при отрицательных долях сплита."""
        with pytest.raises(ValueError, match="должны быть >= 0"):
            RandomDatasetSplitter(val_size=-0.1)

    def test_invalid_too_large_sizes(self):
        """Проверка исключения, если на train не остается данных."""
        with pytest.raises(ValueError, match="должны быть < 1.0"):
            RandomDatasetSplitter(val_size=0.6, test_size=0.5)

    def test_already_split_dataset_skipped(self, dummy_dataset):
        """Если сплиты уже есть, разбиение должно пропуститься."""
        ds = DatasetDict({
            "train": dummy_dataset,
            "validation": dummy_dataset,
            "test": dummy_dataset
        })
        splitter = RandomDatasetSplitter()
        result = splitter(ds)
        
        assert result is ds

    def test_zero_holdout(self, dummy_dataset):
        """Если val_size=0 и test_size=0, возвращается только train (режим индексации)."""
        splitter = RandomDatasetSplitter(val_size=0.0, test_size=0.0)
        result = splitter(dummy_dataset)
        
        assert "validation" not in result
        assert "test" not in result
        assert len(result["train"]) == 100

    def test_only_test_split(self, dummy_dataset):
        """Проверка разбиения, если нужен только test."""
        splitter = RandomDatasetSplitter(val_size=0.0, test_size=0.2, seed=42)
        result = splitter(dummy_dataset)
        
        assert "validation" not in result
        assert len(result["test"]) == 20
        assert len(result["train"]) == 80

    def test_only_val_split(self, dummy_dataset):
        """Проверка разбиения, если нужен только validation."""
        splitter = RandomDatasetSplitter(val_size=0.1, test_size=0.0, seed=42)
        result = splitter(dummy_dataset)
        
        assert "test" not in result
        assert len(result["validation"]) == 10
        assert len(result["train"]) == 90

    def test_full_split(self, dummy_dataset):
        """Проверка полного разбиения на train, val и test (используем доли 0.25 для целой математики)."""
        splitter = RandomDatasetSplitter(val_size=0.25, test_size=0.25, seed=42)
        result = splitter(dummy_dataset)
        
        assert len(result["validation"]) == 25
        assert len(result["test"]) == 25
        assert len(result["train"]) == 50