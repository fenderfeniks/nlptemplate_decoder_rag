from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

# Укажи правильный путь импорта в зависимости от структуры проекта
from src.tools.storage.s3 import S3Storage


# ===========================================================================
# Вспомогательные утилиты для моков boto3
# ===========================================================================


def generate_client_error(code="404", message="Not Found"):
    """Создает реалистичное исключение ClientError от botocore."""
    error_response = {"Error": {"Code": str(code), "Message": message}}
    return ClientError(error_response, "mock_operation")


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def mock_boto_client(mocker):
    """Мокает конструктор boto3.client и возвращает объект-пустышку."""
    mock = mocker.patch("src.tools.storage.s3.boto3.client")
    client_instance = MagicMock()
    mock.return_value = client_instance
    return client_instance


@pytest.fixture
def storage(mock_boto_client):
    """Готовый инстанс S3Storage для тестов."""
    return S3Storage(bucket_name="test-bucket", uri_prefix="s3://", max_concurrency=2)


# ===========================================================================
# Тесты: Одиночные файлы (upload_file / download_file)
# ===========================================================================


class TestS3StorageFiles:
    def test_upload_file_success(self, storage, mock_boto_client, tmp_path):
        """Успешная загрузка файла вызывает upload_file у boto3."""
        local_file = tmp_path / "model.bin"
        local_file.write_text("data")

        storage.upload_file(local_file, "/models/v1/model.bin")

        mock_boto_client.upload_file.assert_called_once_with(
            Filename=str(local_file),
            Bucket="test-bucket",
            Key="models/v1/model.bin",  # Проверяем, что lstrip("/") сработал
            Config=storage.transfer_config,
        )

    def test_upload_file_missing_local(self, storage):
        """Если локального файла нет, метод должен падать до обращения к S3."""
        with pytest.raises(FileNotFoundError, match="Локальный файл не найден"):
            storage.upload_file("non_existent.txt", "remote.txt")

    def test_download_file_atomic(self, storage, mock_boto_client, tmp_path):
        """Скачивание файла использует .tmp и атомарно переименовывает."""
        target_file = tmp_path / "downloaded.bin"

        # Симулируем создание файла при скачивании
        def mock_download_file(Bucket, Key, Filename, Config):
            Path(Filename).write_text("s3_data")

        mock_boto_client.download_file.side_effect = mock_download_file

        result = storage.download_file("remote/model.bin", target_file)

        assert result == target_file
        assert target_file.read_text() == "s3_data"
        assert not target_file.with_name(target_file.name + ".tmp").exists()

        mock_boto_client.download_file.assert_called_once_with(
            "test-bucket",
            "remote/model.bin",
            str(target_file.with_name(target_file.name + ".tmp")),
            Config=storage.transfer_config,
        )

    def test_download_file_cleans_up_on_error(self, storage, mock_boto_client, tmp_path):
        """Если при скачивании S3 упал, локальный .tmp должен удалиться."""
        target_file = tmp_path / "broken.bin"

        def mock_download_file_fail(Bucket, Key, Filename, Config):
            Path(Filename).touch()  # Создали мусор
            raise generate_client_error("403", "Access Denied")

        mock_boto_client.download_file.side_effect = mock_download_file_fail

        with pytest.raises(ClientError):
            storage.download_file("remote.bin", target_file)

        # Проверяем, что мусорного файла нет
        assert not target_file.with_name(target_file.name + ".tmp").exists()


# ===========================================================================
# Тесты: Директории (upload / download)
# ===========================================================================


class TestS3StorageDirectories:
    def test_upload_multithreaded_directory(self, storage, mock_boto_client, tmp_path):
        """Многопоточная выгрузка сканирует файлы локально и грузит пачкой."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "weights.bin").touch()
        (model_dir / "config.json").touch()

        storage.upload(model_dir, "/remote_dir/")

        assert mock_boto_client.upload_file.call_count == 2

        # Собираем переданные параметры (порядок из-за многопоточности не гарантирован)
        uploaded_keys = [kwargs["Key"] for _, kwargs in mock_boto_client.upload_file.call_args_list]
        assert "remote_dir/weights.bin" in uploaded_keys
        assert "remote_dir/config.json" in uploaded_keys

    def test_download_delegates_to_file(self, storage, mock_boto_client, tmp_path):
        """Если head_object не падает (это файл), download проксирует в download_file."""
        target_dir = tmp_path / "model_folder"  # Путь без расширения

        # head_object не кидает исключение, значит S3 думает, что это файл
        mock_boto_client.head_object.return_value = {}

        # Чтобы не писать сложный сайд-эффект, замокаем download_file у самого класса
        storage.download_file = MagicMock(return_value=target_dir / "weights")

        result = storage.download("remote/weights", target_dir)

        # Должен быть склеен путь
        storage.download_file.assert_called_once_with("remote/weights", target_dir / "weights")
        assert result == target_dir / "weights"

    def test_download_paginated_directory(self, storage, mock_boto_client, tmp_path):
        """Если объект - это префикс (404 для файла), скачивается папка через paginator."""
        target_dir = tmp_path / "target_model"

        # 1. head_object кидает 404 (это не единичный файл)
        mock_boto_client.head_object.side_effect = generate_client_error("404")

        # 2. Настраиваем пагинатор list_objects_v2
        mock_paginator = MagicMock()
        mock_boto_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [{"Key": "remote_dir/part1.bin"}, {"Key": "remote_dir/subdir/"}]
            },  # Директорию (заканчивается на /) должен скипнуть
            {"Contents": [{"Key": "remote_dir/part2.bin"}]},
        ]

        def mock_download_file(Bucket, Key, Filename, Config):
            Path(Filename).parent.mkdir(parents=True, exist_ok=True)
            Path(Filename).touch()

        mock_boto_client.download_file.side_effect = mock_download_file

        result = storage.download("remote_dir", target_dir)

        assert result == target_dir
        assert (target_dir / "part1.bin").exists()
        assert (target_dir / "part2.bin").exists()
        assert mock_boto_client.download_file.call_count == 2


# ===========================================================================
# Тесты: Утилиты (exists)
# ===========================================================================


class TestS3StorageExists:
    def test_exists_single_file(self, storage, mock_boto_client):
        """Если head_object успешен — это существующий файл."""
        mock_boto_client.head_object.return_value = {}
        assert storage.exists("remote_model.bin") is True

    def test_exists_prefix_fallback(self, storage, mock_boto_client):
        """Если head_object кидает 404, проверяем наличие ключей через list_objects_v2."""
        mock_boto_client.head_object.side_effect = generate_client_error("404")

        # Симулируем, что list_objects_v2 нашел ключи с таким префиксом
        mock_boto_client.list_objects_v2.return_value = {"Contents": [{"Key": "dir/file"}]}

        assert storage.exists("remote_dir") is True
        mock_boto_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket", Prefix="remote_dir/", MaxKeys=1
        )

    def test_exists_false_when_empty(self, storage, mock_boto_client):
        """Если 404 на файл и пустой префикс — возвращаем False."""
        mock_boto_client.head_object.side_effect = generate_client_error("404")
        mock_boto_client.list_objects_v2.return_value = {}  # Нет ключа Contents

        assert storage.exists("ghost_dir") is False

    def test_exists_raises_on_500(self, storage, mock_boto_client):
        """Если head_object возвращает не 404, а другую ошибку, логируем и возвращаем False."""
        mock_boto_client.head_object.side_effect = generate_client_error("500", "Server Error")

        assert storage.exists("some_path") is False
        mock_boto_client.list_objects_v2.assert_not_called()
