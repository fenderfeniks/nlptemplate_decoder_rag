import pytest

from src.tools.storage.local import LocalStorage


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def storage_env(tmp_path):
    """
    Создает изолированное окружение:
    - storage_dir: эмуляция удаленного хранилища
    - work_dir: эмуляция локальной рабочей директории
    """
    storage_dir = tmp_path / "storage"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    storage = LocalStorage(base_dir=str(storage_dir), uri_prefix="local://")
    return storage, storage_dir, work_dir


class TestLocalStorage:
    # --- Тесты для одиночных файлов (upload_file / download_file) ---

    def test_upload_file_atomic(self, storage_env):
        """Проверка безопасной загрузки одного файла с созданием вложенных папок."""
        storage, storage_dir, work_dir = storage_env
        local_file = work_dir / "config.json"
        local_file.write_text('{"key": "value"}')

        storage.upload_file(local_file, "configs/v1/config.json")

        remote_file = storage_dir / "configs/v1/config.json"
        assert remote_file.exists()
        assert remote_file.read_text() == '{"key": "value"}'
        assert not remote_file.with_suffix(".tmp").exists()

    def test_download_file_success_and_overwrite(self, storage_env):
        """Скачивание файла должно атомарно перезаписывать существующий файл."""
        storage, storage_dir, work_dir = storage_env

        # Подготовка "удаленного" файла
        remote_file = storage_dir / "model.bin"
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.write_text("v2_weights")

        # Подготовка существующего "локального" файла (старая версия)
        local_file = work_dir / "model.bin"
        local_file.write_text("v1_weights")

        storage.download_file("model.bin", local_file)

        assert local_file.read_text() == "v2_weights"
        assert not local_file.with_name(local_file.name + ".tmp").exists()

    def test_download_file_errors(self, storage_env):
        """Ошибки при отсутствии файла или если удаленный путь - директория."""
        storage, storage_dir, work_dir = storage_env
        local_target = work_dir / "target.bin"

        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            storage.download_file("not_exist.bin", local_target)

        # Создаем папку вместо файла
        remote_dir = storage_dir / "models_dir"
        remote_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="Ожидался файл, получена директория"):
            storage.download_file("models_dir", local_target)

    # --- Тесты для директорий (upload / download) ---

    def test_upload_clears_stale_tmp_and_overwrites(self, storage_env):
        """Проверка очистки старых .tmp директорий и атомарной перезаписи."""
        storage, storage_dir, work_dir = storage_env

        # Готовим локальную папку
        source_dir = work_dir / "my_model"
        source_dir.mkdir()
        (source_dir / "weights.bin").write_text("new_data")

        # Симулируем, что в хранилище уже есть старая модель и зависшая .tmp папка от сбоя
        target_dir = storage_dir / "prod_model"
        target_dir.mkdir()
        (target_dir / "weights.bin").write_text("old_data")

        stale_tmp = storage_dir / "prod_model.tmp"
        stale_tmp.mkdir()
        (stale_tmp / "junk").touch()

        storage.upload(source_dir, "prod_model")

        # Проверяем, что .tmp удалена, а целевая папка обновлена
        assert not stale_tmp.exists()
        assert (target_dir / "weights.bin").read_text() == "new_data"

    def test_upload_not_a_directory(self, storage_env):
        """Проверка ошибки, если в upload передали файл вместо директории."""
        storage, _, work_dir = storage_env
        local_file = work_dir / "weights.bin"
        local_file.touch()

        with pytest.raises(NotADirectoryError, match="Ожидалась директория"):
            storage.upload(local_file, "remote_model")

    def test_download_directory_clears_stale_tmp(self, storage_env):
        """Проверка очистки старых .tmp при скачивании директории."""
        storage, storage_dir, work_dir = storage_env

        # Готовим удаленную папку
        remote_dir = storage_dir / "remote_model"
        remote_dir.mkdir()
        (remote_dir / "config.json").write_text("ok")

        # Симулируем локальную зависшую .tmp папку
        target_dir = work_dir / "downloaded_model"
        stale_tmp = work_dir / "downloaded_model.tmp"
        stale_tmp.mkdir()

        storage.download("remote_model", target_dir)

        assert not stale_tmp.exists()
        assert (target_dir / "config.json").read_text() == "ok"

    # --- Тесты хрупкой логики (делегирование в download) ---

    def test_download_delegates_to_file_with_suffix(self, storage_env):
        """Проверка, что скачивание файла из общего метода download работает и добавляет имя к локальной папке."""
        storage, storage_dir, work_dir = storage_env

        remote_file = storage_dir / "dataset.jsonl"
        remote_file.write_text("data")

        # Передаем путь к директории (без суффикса)
        target_dir = work_dir / "datasets"

        storage.download("dataset.jsonl", target_dir)

        # Код должен был склеить target_dir + "dataset.jsonl"
        expected_file = target_dir / "dataset.jsonl"
        assert expected_file.exists()
        assert expected_file.read_text() == "data"

    def test_exists(self, storage_env):
        """Проверка метода exists."""
        storage, storage_dir, _ = storage_env

        (storage_dir / "real_model").mkdir()
        (storage_dir / "real_file.txt").touch()

        assert storage.exists("real_model") is True
        assert storage.exists("real_file.txt") is True
        assert storage.exists("fake_model") is False
