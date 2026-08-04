# tests/pipelines/base/core/data/test_fetcher.py
import os
import pytest
from unittest.mock import patch, MagicMock
import sys
from datasets import Dataset

from src.pipelines.base.core.data.fetcher import (
    _detect_loader,
    RawDataFetcher,
)


class TestDetectLoader:
    def test_detect_loader_standard_extensions(self):
        """Проверка стандартных расширений."""
        assert _detect_loader("data.csv", {}) == ("csv", {})
        assert _detect_loader("data.json", {}) == ("json", {})
        assert _detect_loader("data.parquet", {}) == ("parquet", {})

    def test_detect_loader_glob_patterns(self):
        """Проверка извлечения расширения из glob-паттернов."""
        assert _detect_loader("*.csv", {}) == ("csv", {})
        assert _detect_loader("train-*-of-*.parquet", {}) == ("parquet", {})
        assert _detect_loader("data_?.jsonl", {}) == ("json", {})

    def test_detect_loader_tsv_injection(self):
        """Для tsv должен автоматически добавляться sep='\\t'."""
        loader, kwargs = _detect_loader("data.tsv", {"other_arg": 1})
        assert loader == "csv"
        assert kwargs == {"other_arg": 1, "sep": "\t"}

    def test_detect_loader_unknown_extension(self):
        """Выброс ошибки при неизвестном расширении."""
        with pytest.raises(ValueError, match="Неподдерживаемое расширение"):
            _detect_loader("data.unknown", {})


class TestRawDataFetcherInit:
    def test_invalid_source_type(self, tmp_path):
        """Проверка защиты от неизвестных типов источников."""
        with pytest.raises(ValueError, match="Неизвестный тип источника"):
            RawDataFetcher(source_type="s3", raw_dir=tmp_path)

    @patch.dict(os.environ, {}, clear=True)
    def test_kaggle_missing_env_vars_fail_fast(self, tmp_path):
        """Проверка fail-fast для kaggle (ошибка при инициализации, а не загрузке)."""
        with pytest.raises(EnvironmentError, match="Переменные окружения не установлены"):
            RawDataFetcher(
                source_type="kaggle",
                raw_dir=tmp_path,
                dataset_name="user/dataset",
                file_name="data.csv"
            )


class TestRawDataFetcherLoaders:
    def test_load_local_missing_file_name(self, tmp_path):
        fetcher = RawDataFetcher(source_type="local", raw_dir=tmp_path)
        with pytest.raises(ValueError, match="необходимо указать file_name"):
            fetcher.load()

    def test_load_local_file_not_found(self, tmp_path):
        fetcher = RawDataFetcher(source_type="local", raw_dir=tmp_path, file_name="dummy.csv")
        with pytest.raises(FileNotFoundError, match="Файлы не найдены"):
            fetcher.load()

    @patch("src.pipelines.base.core.data.fetcher.load_dataset")
    def test_load_local_success(self, mock_load_dataset, tmp_path):
        """Проверка успешной загрузки локального файла."""
        # Создаем пустой фейковый файл
        test_file = tmp_path / "test.csv"
        test_file.touch()
        
        fetcher = RawDataFetcher(source_type="local", raw_dir=tmp_path, file_name="*.csv")
        
        mock_dataset = MagicMock(spec=Dataset)
        mock_load_dataset.return_value = mock_dataset
        
        result = fetcher.load()
        
        assert result is mock_dataset
        mock_load_dataset.assert_called_once_with("csv", data_files=[str(test_file)])

    @patch("src.pipelines.base.core.data.fetcher.load_from_disk")
    def test_load_hf_cache_hit(self, mock_load_from_disk, tmp_path):
        """Проверка, что кэш HF датасетов на диске используется."""
        fetcher = RawDataFetcher(source_type="hf", raw_dir=tmp_path, dataset_name="squad")
        
        # Эмулируем существование кэша
        cache_dir = tmp_path / "squad"
        cache_dir.mkdir(parents=True)
        
        mock_load_from_disk.return_value = MagicMock(spec=Dataset)
        result = fetcher.load()
        
        mock_load_from_disk.assert_called_once_with(str(cache_dir))
        assert result is not None

    @patch("src.pipelines.base.core.data.fetcher.load_dataset")
    def test_load_hf_cache_miss(self, mock_load_dataset, tmp_path):
        """Проверка загрузки и сохранения HF датасета при отсутствии кэша."""
        fetcher = RawDataFetcher(
            source_type="hf", 
            raw_dir=tmp_path, 
            dataset_name="squad",
            split="train"
        )
        
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        result = fetcher.load()
        
        mock_load_dataset.assert_called_once_with("squad", token=None, split="train")
        # Проверяем, что кэш сохранен (ключ включает split)
        expected_cache_path = tmp_path / "squad_split-train"
        mock_dataset.save_to_disk.assert_called_once_with(str(expected_cache_path))
        assert result is mock_dataset

class TestRawDataFetcherKaggleAndEdges:
    @patch("src.pipelines.base.core.data.fetcher.load_dataset")
    def test_load_local_multiple_files(self, mock_load, tmp_path):
        """Покрытие ветки else для множества локальных файлов."""
        (tmp_path / "1.csv").touch()
        (tmp_path / "2.csv").touch()
        
        fetcher = RawDataFetcher(source_type="local", raw_dir=tmp_path, file_name="*.csv")
        fetcher.load()
        
        mock_load.assert_called_once()
        assert len(mock_load.call_args[1]["data_files"]) == 2

    @patch.dict(os.environ, {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"})
    def test_kaggle_missing_args(self, tmp_path):
        """Проверка ошибки при отсутствии нужных аргументов Kaggle."""
        fetcher = RawDataFetcher("kaggle", tmp_path, dataset_name=None, file_name=None)
        with pytest.raises(ValueError, match="необходимы dataset_name и file_name"):
            fetcher.load()

    @patch.dict(os.environ, {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"})
    @patch("src.pipelines.base.core.data.fetcher.load_dataset")
    def test_kaggle_local_cache_hit(self, mock_load, tmp_path):
        """Покрытие ветки, где Kaggle датасет уже скачан локально."""
        (tmp_path / "data.csv").touch() # Эмулируем наличие файла
        fetcher = RawDataFetcher("kaggle", tmp_path, dataset_name="a/b", file_name="data.csv")
        fetcher.load()
        mock_load.assert_called_once()

    @patch.dict(os.environ, {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"})
    @patch("src.pipelines.base.core.data.fetcher.load_dataset")
    def test_kaggle_download_triggered(self, mock_load, tmp_path):
        """Покрытие логики скачивания через KaggleApi через подмену sys.modules."""
        # 1. Создаем полностью фейковый модуль и класс
        mock_kaggle_module = MagicMock()
        mock_api_class = MagicMock()
        mock_kaggle_module.KaggleApi = mock_api_class
        
        # 2. Подменяем модуль в системе на время выполнения блока with
        with patch.dict(sys.modules, {'kaggle.api.kaggle_api_extended': mock_kaggle_module}):
            fetcher = RawDataFetcher("kaggle", tmp_path, dataset_name="a/b", file_name="data.csv")
            fetcher.load()
            
            # Проверяем, что класс был инстанцирован и его методы вызваны
            mock_api_instance = mock_api_class.return_value
            mock_api_instance.authenticate.assert_called_once()
            mock_api_instance.dataset_download_files.assert_called_once_with(
                "a/b", path=str(tmp_path), unzip=True
            )
            mock_load.assert_called_once()

    @patch.dict(os.environ, {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"})
    def test_kaggle_import_error(self, tmp_path):
        """Искусственно вызываем ImportError для Kaggle (покрытие except ветки)."""
        # Скрываем модуль kaggle из системных путей
        with patch.dict(sys.modules, {'kaggle.api.kaggle_api_extended': None}):
            fetcher = RawDataFetcher("kaggle", tmp_path, dataset_name="a/b", file_name="data.csv")
            with pytest.raises(ImportError, match="установите: pip install kaggle"):
                fetcher.load()

    def test_hf_missing_dataset_name(self, tmp_path):
        """Покрытие ошибки при отсутствии dataset_name для HF."""
        fetcher = RawDataFetcher("hf", tmp_path, dataset_name=None)
        with pytest.raises(ValueError, match="необходимо указать dataset_name"):
            fetcher.load()