import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class StorageRouter:
    """Маршрутизатор для скачивания артефактов по URI.

    Принимает список инициализированных клиентов и автоматически
    строит маршруты на основе их атрибута `uri_prefix`.
    """

    def __init__(self, clients: list[Any]) -> None:
        # Автоматическая сборка словаря маршрутов: {"s3://": <S3Storage>, ...}
        self.routes = {client.uri_prefix: client for client in clients}

    def _normalize_uri(self, uri: str) -> str:
        """Гарантирует что схема URI имеет двойной слеш: local:/ → local://"""
        import re

        return re.sub(r"^([a-z][a-z0-9+\-.]*):(?!//)", r"\1://", uri)

    def _get_client_and_path(self, uri: str) -> tuple[Any, str]:
        uri = self._normalize_uri(uri)
        for prefix, client in self.routes.items():
            if uri.startswith(prefix):
                remote_path = uri[len(prefix) :].lstrip("/")
                return client, remote_path
        raise ValueError(
            f"Неизвестная схема URI: '{uri}'. Поддерживаемые: {list(self.routes.keys())}"
        )

    def download_from_uri(self, uri: str, cache_dir: Path | str) -> Path:
        client, remote_path = self._get_client_and_path(uri)
        logger.info("StorageRouter: скачивание '%s' через %s", uri, client.__class__.__name__)
        return client.download(remote_path=remote_path, local_dir=cache_dir)

    def download_manifest(self, manifest_uri: str, cache_dir: Path | str) -> dict[str, Any]:
        # Разбиваем URI строкой, чтобы не терять двойной слеш схемы
        last_slash = manifest_uri.rfind("/")
        manifest_filename = manifest_uri[last_slash + 1 :]
        manifest_dir_uri = manifest_uri[:last_slash]

        logger.info("StorageRouter: поиск манифеста '%s'", manifest_uri)
        downloaded_dir = self.download_from_uri(manifest_dir_uri, cache_dir)

        manifest_file = downloaded_dir / manifest_filename
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Файл '{manifest_filename}' не найден в директории {downloaded_dir}"
            )

        with open(manifest_file, encoding="utf-8") as f:
            manifest_data = json.load(f)

        logger.info(
            "Манифест прочитан. Обновлён: %s", manifest_data.get("updated_at", "неизвестно")
        )
        return manifest_data
