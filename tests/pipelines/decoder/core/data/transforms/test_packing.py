# tests/pipelines/decoder/core/data/transforms/test_packing.py
import pytest
from datasets import Dataset

from src.pipelines.decoder.core.data.transforms.packing import SequencePackingTransform


class TestSequencePackingTransform:
    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="должен быть положительным"):
            SequencePackingTransform(packing_chunk_size=-1)

    def test_no_active_columns_skipped(self):
        """Если нет ни одной из нужных колонок, датасет возвращается без изменений."""
        ds = Dataset.from_dict({"text": ["abc"]})
        transform = SequencePackingTransform()
        assert transform(ds) is ds

    def test_partial_missing_columns_warns(self):
        """Если есть только часть колонок, они упаковываются, но логируется warning."""
        ds = Dataset.from_dict({
            "input_ids": [[1, 2], [3, 4]], 
            # labels и attention_mask отсутствуют
        })
        transform = SequencePackingTransform(packing_chunk_size=4, num_proc=1)
        result = transform(ds)
        
        assert len(result) == 1
        assert result["input_ids"] == [[1, 2, 3, 4]]
        assert "attention_mask" not in result.column_names

    def test_packing_drop_remainder(self):
        """Проверка конкатенации и отбрасывания хвоста (drop_remainder=True)."""
        ds = Dataset.from_dict({
            "input_ids": [[1, 2], [3, 4, 5], [6, 7]], 
            "attention_mask": [[1, 1], [1, 1, 1], [1, 1]],
            "labels": [[1, 2], [3, 4, 5], [6, 7]],
        })
        transform = SequencePackingTransform(packing_chunk_size=3, drop_remainder=True, num_proc=1)
        
        result = transform(ds)
        assert len(result) == 2
        assert result["input_ids"] == [[1, 2, 3], [4, 5, 6]]

    def test_packing_keep_remainder(self):
        """Проверка сохранения хвоста (drop_remainder=False)."""
        ds = Dataset.from_dict({
            "input_ids": [[1, 2], [3, 4]], 
            "attention_mask": [[1, 1], [1, 1]],
            "labels": [[1, 2], [3, 4]],
        })
        transform = SequencePackingTransform(packing_chunk_size=3, drop_remainder=False, num_proc=1)
        
        result = transform(ds)
        assert len(result) == 2
        assert result["input_ids"] == [[1, 2, 3], [4]]