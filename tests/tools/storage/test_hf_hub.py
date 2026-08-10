from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.storage.hf_hub import HFHubStorage


class TestHFHubStorage:
    @patch("src.tools.storage.hf_hub.HfApi")
    def test_upload(self, mock_hf_api):
        mock_api_inst = MagicMock()
        mock_hf_api.return_value = mock_api_inst
        storage = HFHubStorage(repo_id="test/repo", uri_prefix="hf://", token="secret")

        fake_path = Path("/fake/dir")

        with patch("pathlib.Path.is_dir", return_value=True):
            storage.upload(local_dir=fake_path, remote_path="models/v1")

        mock_api_inst.upload_folder.assert_called_once_with(
            folder_path=str(fake_path),
            repo_id="test/repo",
            path_in_repo="models/v1",
            repo_type="model",
        )

    @patch("src.tools.storage.hf_hub.snapshot_download")
    @patch("src.tools.storage.hf_hub.shutil")
    def test_download_moves_folder(self, mock_shutil, mock_snapshot):
        storage = HFHubStorage(repo_id="test/repo", uri_prefix="hf://")

        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.rename") as mock_rename,
        ):
            result = storage.download(remote_path="models/v1", local_dir="/target")

            mock_snapshot.assert_called_once()
            assert result == Path("/target")
            mock_rename.assert_called_once_with(Path("/target"))

    @patch("src.tools.storage.hf_hub.HfApi")
    def test_exists_logic(self, mock_hf_api):
        mock_api_inst = MagicMock()
        mock_api_inst.list_repo_files.return_value = [
            "models/v1/config.json",
            "models/v1/pytorch_model.bin",
            "models/v10/config.json",
        ]
        mock_hf_api.return_value = mock_api_inst

        storage = HFHubStorage(repo_id="test/repo", uri_prefix="hf://")

        assert storage.exists("models/v1/config.json") is True
        assert storage.exists("models/v1") is True
        assert storage.exists("models/v1/") is True
        assert storage.exists("models/v2") is False
