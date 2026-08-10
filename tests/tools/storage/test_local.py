import tempfile
from pathlib import Path

import pytest

from src.tools.storage.local import LocalStorage


class TestLocalStorage:
    def test_upload_and_download_atomic(self):
        """Проверка полного цикла локального хранилища с атомарным переименованием."""
        with (
            tempfile.TemporaryDirectory() as storage_dir,
            tempfile.TemporaryDirectory() as work_dir,
        ):
            storage = LocalStorage(base_dir=storage_dir, uri_prefix="local://")

            # 1. Подготовка исходной папки с моделью
            source_dir = Path(work_dir) / "source_model"
            source_dir.mkdir()
            (source_dir / "weights.bin").write_text("dummy_data")

            # 2. Upload
            storage.upload(local_dir=source_dir, remote_path="my_model_v1")

            # Проверяем, что файл появился в хранилище, а .tmp папки нет
            remote_full_path = Path(storage_dir) / "my_model_v1"
            assert remote_full_path.exists()
            assert (remote_full_path / "weights.bin").read_text() == "dummy_data"
            assert not Path(storage_dir, "my_model_v1.tmp").exists()
            assert storage.exists("my_model_v1")

            # 3. Download
            target_dir = Path(work_dir) / "downloaded_model"
            result_path = storage.download(remote_path="my_model_v1", local_dir=target_dir)

            assert result_path == target_dir
            assert (target_dir / "weights.bin").read_text() == "dummy_data"
            assert not Path(work_dir, "downloaded_model.tmp").exists()

    def test_upload_not_a_directory(self):
        """Upload должен падать, если передали путь к файлу вместо папки."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = LocalStorage(base_dir=tmp_dir, uri_prefix="local://")
            file_path = Path(tmp_dir) / "file.txt"
            file_path.touch()

            with pytest.raises(NotADirectoryError):
                storage.upload(local_dir=file_path, remote_path="model")
