# tests/decoder_pipeline/core/data/test_fetcher.py
from pathlib import Path
from unittest.mock import patch

import pytest

from src.decoder_pipeline.core.data.fetcher import RawDataFetcher, _detect_loader


class TestDetectLoader:
    def test_detects_supported_extensions(self):
        loader, kwargs = _detect_loader("data.csv", {})
        assert loader == "csv"
        assert kwargs == {}

    def test_detects_tsv_and_adds_separator(self):
        loader, kwargs = _detect_loader("data.tsv", {})
        assert loader == "csv"
        assert kwargs.get("sep") == "\t"

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Неподдерживаемое расширение файла"):
            _detect_loader("archive.zip", {})


class TestRawDataFetcher:
    """
    def test_unknown_source_type_raises(self, tmp_path):
        fetcher = RawDataFetcher(source_type="unknown", raw_dir=tmp_path)
        with pytest.raises(ValueError, match="Неизвестный тип источника данных"):
            fetcher.load()"""

    def test_local_missing_file_name_raises(self, tmp_path):
        fetcher = RawDataFetcher(source_type="local", raw_dir=tmp_path, file_name=None)
        with pytest.raises(ValueError, match="необходимо указать file_name"):
            fetcher.load()
    """
    def test_local_file_not_found_raises(self, tmp_path):
        fetcher = RawDataFetcher(source_type="local", raw_dir=tmp_path, file_name="missing.csv")
        with pytest.raises(FileNotFoundError):
            fetcher.load()"""

    @patch("src.decoder_pipeline.core.data.fetcher.load_dataset")
    def test_local_loads_successfully(self, mock_load_dataset, tmp_path):
        csv_file = tmp_path / "train.csv"
        csv_file.write_text("col1,col2\nval1,val2")

        fetcher = RawDataFetcher(source_type="local", raw_dir=tmp_path, file_name="train.csv")
        fetcher.load()

        mock_load_dataset.assert_called_once()
        _, kwargs = mock_load_dataset.call_args
        assert str(csv_file) in kwargs["data_files"][0]