# tests/pipelines/base/core/data/transforms/test_filtering.py
import pytest
from datasets import Dataset

from src.pipelines.base.core.data.transforms.filtering import LengthFilterTransform


class TestLengthFilterTransform:
    def test_invalid_max_length(self):
        """Проверка выброса ошибки при отрицательной или нулевой длине."""
        with pytest.raises(ValueError, match="должен быть положительным числом"):
            LengthFilterTransform(max_length=0)
            
        with pytest.raises(ValueError):
            LengthFilterTransform(max_length=-5)

    def test_length_filtering(self, sample_tokenized_dataset: Dataset):
        """Проверка корректной фильтрации последовательностей."""
        transform = LengthFilterTransform(max_length=5, column="input_ids", num_proc=1)
        result = transform(sample_tokenized_dataset)
        
        assert len(result) == 2
        assert result["text"] == ["short", "very short"]
        
    def test_missing_column_warning(self, sample_tokenized_dataset: Dataset):
        """Проверка поведения при отсутствии целевой колонки."""
        transform = LengthFilterTransform(max_length=5, column="missing_ids")
        result = transform(sample_tokenized_dataset)
        
        # Датасет не должен измениться
        assert len(result) == len(sample_tokenized_dataset)