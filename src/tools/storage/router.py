import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class StorageRouter:
    """Маршрутизатор для скачивания артефактов по URI (s3://, hf://, local://).

    Таблица маршрутизации строится из явных пар (uri_prefix → клиент),
    переданных в конструктор. Клиенты-хранилища ничего не знают о своих
    URI-префиксах — это ответственность роутера.
    """

    def __init__(
        self,
        routes: dict[str, Any],
    ) -> None:
        """
        Args:
            routes: Словарь {uri_prefix: client}, например:
                {
                    "s3://":     <S3Storage>,
                    "hf://":     <HFHubStorage>,
                    "local://":  <LocalStorage>,
                }
        """
        self.routes = routes

    def _get_client_and_path(self, uri: str) -> tuple[Any, str]:
        for prefix, client in self.routes.items():
            if uri.startswith(prefix):
                if client is None:
                    raise RuntimeError(f"Клиент для схемы '{prefix}' не передан в StorageRouter.")
                remote_path = uri[len(prefix) :].lstrip("/")
                return client, remote_path

        raise ValueError(
            f"Неизвестная схема URI: '{uri}'. Поддерживаемые: {list(self.routes.keys())}"
        )

    def download_from_uri(self, uri: str, cache_dir: Path | str) -> Path:
        """Скачивает артефакт по URI в локальную папку."""
        client, remote_path = self._get_client_and_path(uri)
        logger.info("StorageRouter: скачивание '%s' через %s", uri, client.__class__.__name__)
        return client.download(remote_path=remote_path, local_dir=cache_dir)

    def download_manifest(self, manifest_uri: str, cache_dir: Path | str) -> dict[str, Any]:
        """Скачивает JSON-манифест и возвращает его как словарь."""
        uri_path = Path(manifest_uri)
        manifest_filename = uri_path.name
        manifest_dir_uri = str(uri_path.parent).replace("\\", "/")

        logger.info("StorageRouter: поиск манифеста '%s'", manifest_uri)

        downloaded_dir = self.download_from_uri(manifest_dir_uri, cache_dir)

        manifest_file = downloaded_dir / manifest_filename
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Файл '{manifest_filename}' не найден в скачанной директории {downloaded_dir}"
            )

        with open(manifest_file, encoding="utf-8") as f:
            manifest_data = json.load(f)

        logger.info(
            "Манифест прочитан. Обновлён: %s", manifest_data.get("updated_at", "неизвестно")
        )
        return manifest_data
