from abc import ABC, abstractmethod
from pathlib import Path


class BaseStorage(ABC):
    """Абстрактный интерфейс для работы с хранилищами артефактов."""

    def __init__(self, uri_prefix: str) -> None:
        self.uri_prefix = uri_prefix

    @abstractmethod
    def upload(self, local_dir: Path | str, remote_path: str) -> None:
        """Загружает локальную директорию в удаленное хранилище.

        Args:
            local_dir: Путь к локальной папке с моделью.
            remote_path: Путь/префикс назначения в хранилище.
        """
        pass

    @abstractmethod
    def download(self, remote_path: str, local_dir: Path | str) -> Path:
        """Скачивает модель из хранилища в локальную директорию.

        Args:
            remote_path: Путь/префикс в хранилище.
            local_dir: Локальная целевая папка.

        Returns:
            Path: Путь к готовой локальной директории с весами.
        """
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: Path | str) -> Path:
        """Скачивает один файл из хранилища.
        
        Args:
            remote_path: Путь к файлу в хранилище.
            local_path:  Локальный путь куда сохранить файл.
        
        Returns:
            Path к скачанному файлу.
        """
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Проверяет, существует ли директория или файл в удаленном хранилище."""
        pass
