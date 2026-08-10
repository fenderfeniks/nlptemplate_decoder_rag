import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.tools.storage.router import StorageRouter


class TestStorageRouter:
    def test_uri_normalization(self):
        router = StorageRouter([])
        assert router._normalize_uri("s3://bucket/path") == "s3://bucket/path"
        # ИСПРАВЛЕНИЕ: Ожидаем 3 слэша, так как код сохраняет оригинальный слэш пути
        assert router._normalize_uri("local:/opt/models") == "local:///opt/models"

    def test_routing_logic(self):
        s3_client = MagicMock(uri_prefix="s3://")
        hf_client = MagicMock(uri_prefix="hf://")
        local_client = MagicMock(uri_prefix="local://")

        router = StorageRouter(clients=[s3_client, hf_client, local_client])

        client, path = router._get_client_and_path("s3://my_models/v1")
        assert client == s3_client
        assert path == "my_models/v1"

        with pytest.raises(ValueError, match="Неизвестная схема URI"):
            router._get_client_and_path("gcp://models")

    def test_download_manifest(self):
        mock_client = MagicMock(uri_prefix="local://")

        def mock_download(remote_path, local_dir):
            # Эмулируем создание скачанного манифеста в папке кэша
            manifest_file = Path(local_dir) / "manifest.json"
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_file, "w") as f:
                json.dump({"load_type": "lora", "updated_at": "2026-08-10"}, f)
            return Path(local_dir)

        mock_client.download.side_effect = mock_download
        router = StorageRouter(clients=[mock_client])

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_data = router.download_manifest(
                manifest_uri="local://storage/manifests/manifest.json", cache_dir=tmp_dir
            )

            assert manifest_data["load_type"] == "lora"
            assert manifest_data["updated_at"] == "2026-08-10"
            mock_client.download.assert_called_once_with(
                remote_path="storage/manifests", local_dir=tmp_dir
            )
