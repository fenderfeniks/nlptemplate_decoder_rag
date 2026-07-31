# tests/decoder_pipeline/core/data/test_splitters.py
from datasets import Dataset, DatasetDict

from src.decoder_pipeline.core.data.splitters import RandomDatasetSplitter


def _make_dummy_dataset(size: int = 100) -> Dataset:
    return Dataset.from_dict({"id": list(range(size))})


class TestRandomDatasetSplitter:
    def test_splits_into_three_parts(self):
        ds = _make_dummy_dataset(100)
        splitter = RandomDatasetSplitter(val_size=0.1, test_size=0.1, seed=42)
        result = splitter(ds)

        assert isinstance(result, DatasetDict)
        assert set(result.keys() & {"train", "validation", "test"}) == {"train", "validation", "test"}
        assert len(result["train"]) == 80
        assert len(result["validation"]) == 10
        assert len(result["test"]) == 10

    def test_zero_holdout_returns_only_train(self):
        ds = _make_dummy_dataset(50)
        splitter = RandomDatasetSplitter(val_size=0.0, test_size=0.0)
        result = splitter(ds)

        assert list(result.keys()) == ["train"]
        assert len(result["train"]) == 50

    def test_pre_existing_splits_are_preserved(self):
        ds_dict = DatasetDict({
            "train": _make_dummy_dataset(10),
            "validation": _make_dummy_dataset(5),
            "test": _make_dummy_dataset(5),
        })
        splitter = RandomDatasetSplitter()
        result = splitter(ds_dict)
        assert result == ds_dict