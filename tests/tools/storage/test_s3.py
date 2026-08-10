import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tools.storage.s3 import S3Storage


class TestS3Storage:
    @patch("src.tools.storage.s3.boto3.client")
    def test_upload_multithreaded(self, mock_boto):
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        storage = S3Storage(bucket_name="my-bucket", uri_prefix="s3://", max_concurrency=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "model.bin").touch()
            (tmp_path / "config.json").touch()

            storage.upload(local_dir=tmp_path, remote_path="v1")

            # Проверяем, что upload_file был вызван для обоих файлов
            assert mock_client.upload_file.call_count == 2
            uploaded_keys = [call[0][2] for call in mock_client.upload_file.call_args_list]
            assert "v1/model.bin" in uploaded_keys
            assert "v1/config.json" in uploaded_keys

    @patch("src.tools.storage.s3.boto3.client")
    def test_exists(self, mock_boto):
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        storage = S3Storage(bucket_name="my-bucket", uri_prefix="s3://")

        # Возвращаем имитацию ответа API с объектами
        mock_client.list_objects_v2.return_value = {"Contents": [{"Key": "v1/model.bin"}]}
        assert storage.exists("v1") is True

        # Пустой ответ API
        mock_client.list_objects_v2.return_value = {}
        assert storage.exists("v2") is False
