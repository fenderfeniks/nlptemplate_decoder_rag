import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class StorageRouter:
    """Маршрутизатор для скачивания артефактов по URI."""

    def __init__(self, clients: list[Any]) -> None:
        self.routes = {client.uri_prefix: client for client in clients}

    def _normalize_uri(self, uri: str) -> str:
        """Гарантирует что схема URI имеет двойной слеш: local:/ -> local://"""
        import re
        return re.sub(r"^([a-z][a-z0-9+\-.]*):(?!//)", r"\1://", uri)

    def _get_client_and_path(self, uri: str) -> tuple[Any, str]:
        uri = self._normalize_uri(uri)
        for prefix, client in self.routes.items():
            if uri.startswith(prefix):
                remote_path = uri[len(prefix):].lstrip("/")
                return client, remote_path
        raise ValueError(
            f"Неизвестная схема URI: '{uri}'. Поддерживаемые: {list(self.routes.keys())}"
        )

    def download_from_uri(self, uri: str, cache_dir: Path | str) -> Path:
        """Скачивает директорию из хранилища."""
        client, remote_path = self._get_client_and_path(uri)
        logger.info(
            "StorageRouter: скачивание '%s' через %s",
            uri, client.__class__.__name__,
        )
        return client.download(remote_path=remote_path, local_dir=cache_dir)

    def download_file_from_uri(self, uri: str, local_path: Path | str) -> Path:
        """Скачивает один файл из хранилища."""
        client, remote_path = self._get_client_and_path(uri)
        logger.info(
            "StorageRouter: скачивание файла '%s' через %s",
            uri, client.__class__.__name__,
        )
        return client.download_file(remote_path=remote_path, local_path=local_path)

    def upload_file_to_uri(self, local_path: Path | str, uri: str) -> None:
        """Загружает один файл в хранилище по URI.

        Зеркало download_file_from_uri — используется для обновления манифеста
        и загрузки одиночных артефактов (GGUF, JSON и т.п.).

        Args:
            local_path: Путь к локальному файлу.
            uri:        URI назначения, например local://decoder_pipeline/model.gguf
        """
        client, remote_path = self._get_client_and_path(uri)
        logger.info(
            "StorageRouter: загрузка файла '%s' → '%s' через %s",
            local_path, uri, client.__class__.__name__,
        )
        client.upload_file(local_path=local_path, remote_path=remote_path)

    def upload_dir_to_uri(self, local_dir: Path | str, uri: str) -> None:
        """Загружает директорию в хранилище по URI.

        Зеркало download_from_uri — атомарная загрузка через tmp + rename
        (логика в LocalStorage.upload).

        Args:
            local_dir: Путь к локальной директории.
            uri:       URI назначения, например local://decoder_pipeline/merged_model
        """
        client, remote_path = self._get_client_and_path(uri)
        logger.info(
            "StorageRouter: загрузка директории '%s' → '%s' через %s",
            local_dir, uri, client.__class__.__name__,
        )
        client.upload(local_dir=local_dir, remote_path=remote_path)

    def download_manifest(self, manifest_uri: str, cache_dir: Path | str) -> dict[str, Any]:
        """Скачивает конкретный файл манифеста."""
        logger.info("StorageRouter: поиск манифеста '%s'", manifest_uri)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_filename = manifest_uri.split("/")[-1]
        local_path = cache_dir / manifest_filename

        downloaded_file = self.download_file_from_uri(manifest_uri, local_path)

        if not downloaded_file.exists():
            raise FileNotFoundError(f"Файл '{manifest_filename}' не найден по пути {downloaded_file}")

        with open(downloaded_file, encoding="utf-8") as f:
            manifest_data = json.load(f)

        logger.info(
            "Манифест прочитан. Обновлён: %s", manifest_data.get("updated_at", "неизвестно")
        )
        return manifest_data