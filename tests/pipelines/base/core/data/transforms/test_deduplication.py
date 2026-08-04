# tests/pipelines/base/core/data/transforms/test_deduplication.py
import pytest
from unittest.mock import patch
from datasets import Dataset

from src.pipelines.base.core.data.transforms.deduplication import (
    ExactDeduplicationTransform,
    MinHashDeduplicationTransform,
)

class TestExactDeduplicationTransform:
    def test_empty_target_columns_raises_error(self):
        with pytest.raises(ValueError, match="target_columns не может быть пустым"):
            ExactDeduplicationTransform(target_columns=[])

    def test_exact_deduplication_single_column(self, sample_text_dataset: Dataset):
        transform = ExactDeduplicationTransform(target_columns=["text"], num_proc=1)
        result = transform(sample_text_dataset)
        
        assert len(result) == 3
        assert result["text"] == ["Пример текста 1", "Пример текста 2", "пример текста 1 "]

    def test_exact_deduplication_missing_column(self, sample_text_dataset: Dataset):
        transform = ExactDeduplicationTransform(target_columns=["missing_col"])
        result = transform(sample_text_dataset)
        assert len(result) == len(sample_text_dataset)


class TestMinHashDeduplicationTransform:
    def test_empty_target_columns_raises_error(self):
        with pytest.raises(ValueError):
            MinHashDeduplicationTransform(target_columns=[])

    def test_import_error_if_no_datasketch(self):
        """Искусственно убираем MinHash, чтобы проверить выброс исключения."""
        with patch("src.pipelines.base.core.data.transforms.deduplication.MinHash", None):
            with pytest.raises(ImportError, match="установите: pip install datasketch"):
                MinHashDeduplicationTransform(target_columns=["text"])

    def test_minhash_missing_column(self, sample_text_dataset: Dataset):
        """Если колонка отсутствует, датасет должен вернуться без изменений."""
        try:
            transform = MinHashDeduplicationTransform(target_columns=["missing"])
        except ImportError:
            pytest.skip("datasketch не установлен")
            
        result = transform(sample_text_dataset)
        assert len(result) == len(sample_text_dataset)

    def test_minhash_short_text_branch(self):
        """Проверка ветки, где текст короче размера шингла (ngram_size)."""
        ds = Dataset.from_dict({"text": ["коротко", "коротко"]})
        try:
            transform = MinHashDeduplicationTransform(
                target_columns=["text"], 
                ngram_size=5,  # ngram больше чем слов в тексте (1)
                num_proc=1
            )
        except ImportError:
            pytest.skip("datasketch не установлен")
            
        result = transform(ds)
        # Должен остаться только один элемент (дубликат удален)
        assert len(result) == 1

    def test_minhash_deduplication(self, sample_text_dataset: Dataset):
        try:
            transform = MinHashDeduplicationTransform(
                target_columns=["text"], 
                threshold=0.5,
                ngram_size=2,
                num_proc=1
            )
        except ImportError:
            pytest.skip("datasketch не установлен")
            
        result = transform(sample_text_dataset)
        assert len(result) == 2
        assert result["text"] == ["Пример текста 1", "Пример текста 2"]