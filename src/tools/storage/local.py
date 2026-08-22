import logging
import shutil
from pathlib import Path

from src.tools.storage.base import BaseStorage


logger = logging.getLogger(__name__)


class LocalStorage(BaseStorage):
    """Реализация хранилища для локальной файловой системы."""

    def __init__(self, base_dir: str, uri_prefix: str) -> None:
        super().__init__(uri_prefix=uri_prefix)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        """Безопасная загрузка одного файла без удаления остального содержимого папки."""
        local_path = Path(local_path)
        dest_path = Path(self.base_dir) / remote_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Атомарная перезапись файла (сначала копируем во временный, потом переименовываем)
        tmp_dest = dest_path.with_suffix(".tmp")
        shutil.copy2(local_path, tmp_dest)
        tmp_dest.replace(dest_path)

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

        # --- ДОБАВЛЕННЫЙ БЛОК ---
        # Если источник это файл (например, benchmark.jsonl),
        # делегируем задачу специализированному методу скачивания файлов.
        if source_path.is_file():
            # Если целевой путь (target_path) это директория (как ожидалось при скачивании артефактов),
            # добавляем к нему имя файла. Если это уже полный путь к файлу - оставляем.
            if target_path.suffix == "":
                target_path = target_path / source_path.name
            return self.download_file(remote_path, target_path)
        # ------------------------

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

    def download_file(self, remote_path: str, local_path: Path | str) -> Path:
        source = self.base_dir / remote_path
        target = Path(local_path)

        if not source.exists():
            raise FileNotFoundError(f"Файл не найден в хранилище: {source}")
        if not source.is_file():
            raise ValueError(f"Ожидался файл, получена директория: {source}")

        tmp = target.with_name(target.name + ".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, tmp)
        if target.exists():
            target.unlink()
        tmp.rename(target)

        logger.info("Файл атомарно скачан: %s -> %s", source, target)
        return target

    def exists(self, remote_path: str) -> bool:
        target_path = self.base_dir / remote_path
        return target_path.exists()
