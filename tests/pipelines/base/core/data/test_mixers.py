# tests/pipelines/base/core/data/test_mixers.py
import pytest
from unittest.mock import patch, MagicMock

from datasets import Dataset, DatasetDict

from src.pipelines.base.core.data.mixers import InterleavedDataFetcher


class TestInterleavedDataFetcherInit:
    def test_length_mismatch(self):
        """Количество fetchers должно совпадать с probabilities."""
        with pytest.raises(ValueError, match="должно совпадать с числом probabilities"):
            InterleavedDataFetcher(fetchers=[1, 2], probabilities=[1.0])

    def test_invalid_probabilities(self):
        """Сумма вероятностей должна быть равна 1, и они не могут быть нулевыми."""
        with pytest.raises(ValueError, match="Все вероятности должны быть > 0"):
            InterleavedDataFetcher(fetchers=[1, 2], probabilities=[0.0, 1.0])
            
        with pytest.raises(ValueError, match="Сумма probabilities должна быть равна 1.0"):
            InterleavedDataFetcher(fetchers=[1, 2], probabilities=[0.5, 0.4])

    def test_invalid_stopping_strategy(self):
        """Защита от опечаток в stopping_strategy."""
        with pytest.raises(ValueError, match="Неизвестная stopping_strategy"):
            InterleavedDataFetcher(
                fetchers=[1], 
                probabilities=[1.0], 
                stopping_strategy="unknown"
            )


class TestInterleavedDataFetcherLoad:
    @patch("src.pipelines.base.core.data.mixers.interleave_datasets")
    def test_successful_mix(self, mock_interleave):
        """Успешное смешивание нескольких источников."""
        # Мокаем два загрузчика: один возвращает DatasetDict, другой простой Dataset
        ds1 = Dataset.from_dict({"text": ["a", "b"]})
        ds2 = Dataset.from_dict({"text": ["c", "d"]})
        
        mock_fetcher1 = MagicMock()
        mock_fetcher1.load.return_value = DatasetDict({"train": ds1, "test": ds1})
        
        mock_fetcher2 = MagicMock()
        mock_fetcher2.load.return_value = ds2
        
        mixed_result = Dataset.from_dict({"text": ["a", "c", "b", "d"]})
        mock_interleave.return_value = mixed_result
        
        mixer = InterleavedDataFetcher(
            fetchers=[mock_fetcher1, mock_fetcher2],
            probabilities=[0.7, 0.3],
            seed=123,
            stopping_strategy="all_exhausted"
        )
        
        result = mixer.load()
        
        # interleave_datasets должен получить именно train-сплиты
        mock_interleave.assert_called_once_with(
            [ds1, ds2],
            probabilities=[0.7, 0.3],
            seed=123,
            stopping_strategy="all_exhausted"
        )
        
        # Миксер должен обернуть результат обратно в DatasetDict с ключом 'train'
        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert result["train"] is mixed_result

    def test_missing_train_split(self):
        """Если у одного из источников нет train-сплита, должна быть ошибка."""
        mock_fetcher = MagicMock()
        # Возвращаем DatasetDict только с test-сплитом
        mock_fetcher.load.return_value = DatasetDict({"test": Dataset.from_dict({"t": [1]})})
        
        mixer = InterleavedDataFetcher(fetchers=[mock_fetcher], probabilities=[1.0])
        
        with pytest.raises(ValueError, match="не содержит сплита 'train'"):
            mixer.load()