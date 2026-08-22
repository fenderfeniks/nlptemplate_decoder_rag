import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Укажи правильный путь импорта
from src.tools.storage.router import StorageRouter


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def mock_clients():
    """Создает моки для клиентов различных хранилищ."""
    s3_client = MagicMock()
    s3_client.uri_prefix = "s3://"

    local_client = MagicMock()
    local_client.uri_prefix = "local://"

    hf_client = MagicMock()
    hf_client.uri_prefix = "hf://"

    return [s3_client, local_client, hf_client]


@pytest.fixture
def router(mock_clients):
    """Готовый инстанс StorageRouter с замоканными клиентами."""
    return StorageRouter(clients=mock_clients)


# ===========================================================================
# Тесты парсинга и маршрутизации
# ===========================================================================


class TestStorageRouterCore:
    def test_uri_normalization(self, router):
        """Проверка восстановления корректной схемы URI."""
        # Уже нормализованные
        assert router._normalize_uri("s3://bucket/path") == "s3://bucket/path"

        # Требующие нормализации (один слэш после двоеточия)
        # local:/opt/models -> local:// + /opt/models -> local:///opt/models
        assert router._normalize_uri("local:/opt/models") == "local:///opt/models"
        assert router._normalize_uri("hf:/org/model") == "hf:///org/model"

    def test_get_client_and_path_success(self, router, mock_clients):
        """Проверка успешного матчинга префиксов и извлечения remote_path."""
        s3_client, local_client, hf_client = mock_clients

        # S3
        client, path = router._get_client_and_path("s3://my-bucket/models/v1")
        assert client == s3_client
        assert path == "my-bucket/models/v1"

        # Local (слэши в начале пути отрезаются через lstrip)
        client, path = router._get_client_and_path("local:///mnt/data/model")
        assert client == local_client
        assert path == "mnt/data/model"

    def test_get_client_and_path_unknown_scheme(self, router):
        """Ошибка при использовании незарегистрированной схемы URI."""
        with pytest.raises(ValueError, match="Неизвестная схема URI: 'gcp://bucket'"):
            router._get_client_and_path("gcp://bucket/path")


# ===========================================================================
# Тесты делегирования (проксирования вызовов к клиентам)
# ===========================================================================


class TestStorageRouterDelegation:
    def test_download_from_uri(self, router, mock_clients):
        s3_client = mock_clients[0]
        s3_client.download.return_value = Path("/local/cache/dir")

        result = router.download_from_uri("s3://bucket/model_dir", "/local/cache/dir")

        s3_client.download.assert_called_once_with(
            remote_path="bucket/model_dir", local_dir="/local/cache/dir"
        )
        assert result == Path("/local/cache/dir")

    def test_download_file_from_uri(self, router, mock_clients):
        local_client = mock_clients[1]
        local_client.download_file.return_value = Path("/local/file.txt")

        result = router.download_file_from_uri("local:///opt/data/file.txt", "/local/file.txt")

        local_client.download_file.assert_called_once_with(
            remote_path="opt/data/file.txt", local_path="/local/file.txt"
        )
        assert result == Path("/local/file.txt")

    def test_upload_file_to_uri(self, router, mock_clients):
        hf_client = mock_clients[2]

        router.upload_file_to_uri("/local/weights.bin", "hf://org/repo/weights.bin")

        hf_client.upload_file.assert_called_once_with(
            local_path="/local/weights.bin", remote_path="org/repo/weights.bin"
        )

    def test_upload_dir_to_uri(self, router, mock_clients):
        s3_client = mock_clients[0]

        router.upload_dir_to_uri("/local/model_dir", "s3://bucket/remote_dir")

        s3_client.upload.assert_called_once_with(
            local_dir="/local/model_dir", remote_path="bucket/remote_dir"
        )


# ===========================================================================
# Тесты бизнес-логики манифеста
# ===========================================================================


class TestStorageRouterManifest:
    def test_download_manifest_success(self, router, mock_clients, tmp_path):
        """Успешное скачивание и парсинг манифеста."""
        s3_client = mock_clients[0]

        # Настраиваем сайд-эффект для download_file_from_uri
        def mock_download_file(remote_path, local_path):
            # Физически создаем файл манифеста, имитируя успешное скачивание
            path_obj = Path(local_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text('{"version": "1.0", "updated_at": "2026-08-22"}')
            return path_obj

        s3_client.download_file.side_effect = mock_download_file

        manifest_data = router.download_manifest("s3://bucket/manifests/prod.json", tmp_path)

        # Проверки логики
        assert manifest_data["version"] == "1.0"
        assert manifest_data["updated_at"] == "2026-08-22"
        s3_client.download_file.assert_called_once_with(
            remote_path="bucket/manifests/prod.json", local_path=tmp_path / "prod.json"
        )

    def test_download_manifest_file_not_found(self, router, mock_clients, tmp_path):
        """Если скачанный файл физически не появился на диске, выбрасывается FileNotFoundError."""
        s3_client = mock_clients[0]

        # Имитируем скачивание, которое возвращает путь, но не создает файл
        s3_client.download_file.return_value = tmp_path / "prod.json"

        with pytest.raises(FileNotFoundError, match="Файл 'prod.json' не найден"):
            router.download_manifest("s3://bucket/manifests/prod.json", tmp_path)

    def test_download_manifest_invalid_json(self, router, mock_clients, tmp_path):
        """Если скачанный манифест содержит битый JSON, это должно обрабатываться штатно."""
        s3_client = mock_clients[0]

        def mock_download_file(remote_path, local_path):
            path_obj = Path(local_path)
            path_obj.write_text('{"broken_json: }')
            return path_obj

        s3_client.download_file.side_effect = mock_download_file

        with pytest.raises(json.JSONDecodeError):
            router.download_manifest("s3://bucket/manifest.json", tmp_path)
