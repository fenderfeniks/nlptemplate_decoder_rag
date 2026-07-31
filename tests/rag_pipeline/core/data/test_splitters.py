# tests/rag_pipeline/core/data/test_splitters.py
import pytest
from datasets import Dataset, DatasetDict

from src.rag_pipeline.core.data.splitters import RandomDatasetSplitter


def _make_dummy_dataset(size: int = 100) -> Dataset:
    return Dataset.from_dict({"id": list(range(size))})


class TestRandomDatasetSplitter:
    def test_standard_split(self):
        ds = _make_dummy_dataset(100)
        splitter = RandomDatasetSplitter(val_size=0.1, test_size=0.1, seed=42)
        res = splitter(ds)
        
        assert isinstance(res, DatasetDict)
        assert len(res["train"]) == 80
        assert len(res["validation"]) == 10
        assert len(res["test"]) == 10

    def test_zero_holdout_returns_only_train(self):
        """Если val и test равны нулю, возвращаем только train."""
        ds = _make_dummy_dataset(100)
        splitter = RandomDatasetSplitter(val_size=0.0, test_size=0.0)
        res = splitter(ds)
        
        assert list(res.keys()) == ["train"]
        assert len(res["train"]) == 100

    def test_invalid_proportions_raise(self):
        with pytest.raises(ValueError):
            RandomDatasetSplitter(val_size=0.6, test_size=0.5) # Сумма > 1.0