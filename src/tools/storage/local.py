import logging
import shutil
from pathlib import Path

from src.tools.storage.base import BaseStorage


logger = logging.getLogger(__name__)


class LocalStorage(BaseStorage):
    """Реализация хранилища для локальной файловой системы с атомарными операциями."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, local_dir: Path | str, remote_path: str) -> None:
        source_path = Path(local_dir)
        if not source_path.is_dir():
            raise NotADirectoryError(f"Ожидалась директория, получено: {source_path}")

        target_path = self.base_dir / remote_path
        tmp_path = target_path.with_name(target_path.name + ".tmp")

        # Очищаем временную папку, если она осталась от прошлого сбоя
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        logger.debug("Копирование во временную директорию: %s", tmp_path)
        shutil.copytree(source_path, tmp_path)

        # Атомарная подмена
        if target_path.exists():
            shutil.rmtree(target_path)
        tmp_path.rename(target_path)

        logger.info("Модель атомарно сохранена в локальное хранилище: %s", target_path)

    def download(self, remote_path: str, local_dir: Path | str) -> Path:
        source_path = self.base_dir / remote_path
        target_path = Path(local_dir)

        if not source_path.exists():
            raise FileNotFoundError(f"Модель не найдена в хранилище: {source_path}")

        tmp_path = target_path.with_name(target_path.name + ".tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        logger.debug("Копирование из кэша во временную директорию: %s", tmp_path)
        shutil.copytree(source_path, tmp_path)

        if target_path.exists():
            shutil.rmtree(target_path)
        tmp_path.rename(target_path)

        logger.info("Модель атомарно загружена из хранилища в: %s", target_path)
        return target_path
