import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.tools.storage.base import BaseStorage


logger = logging.getLogger(__name__)


class HFHubStorage(BaseStorage):
    """Хранилище Hugging Face Hub с защитой от битых кэшей и симлинков."""

    def __init__(self, repo_id: str, token: str | None = None, repo_type: str = "model") -> None:
        self.repo_id = repo_id
        self.token = token
        self.repo_type = repo_type
        self.api = HfApi(token=self.token)

    def upload(self, local_dir: Path | str, remote_path: str) -> None:
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise NotADirectoryError(f"Путь {local_path} должен быть директорией.")

        remote_path = remote_path.strip("/")

        logger.info("Загрузка папки в HF Hub репозиторий: %s", self.repo_id)
        self.api.upload_folder(
            folder_path=str(local_path),
            repo_id=self.repo_id,
            path_in_repo=remote_path,
            repo_type=self.repo_type,
        )
        logger.info("Модель успешно загружена в HF Hub (путь: %s)", remote_path)

    def download(self, remote_path: str, local_dir: Path | str) -> Path:
        target_path = Path(local_dir)
        remote_path = remote_path.strip("/")
        tmp_path = target_path.with_name(target_path.name + ".tmp")

        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        logger.info("Скачивание файлов из HF Hub: %s/%s", self.repo_id, remote_path)

        try:
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=str(tmp_path),
                allow_patterns=[f"{remote_path}/*"],
                token=self.token,
                repo_type=self.repo_type,
                local_dir_use_symlinks=False,  # Выгружаем физические файлы
            )

            # snapshot_download сохраняет иерархию репозитория внутри local_dir.
            # Нам нужно достать файлы из подпапки remote_path.
            extracted_path = tmp_path / remote_path

            if target_path.exists():
                shutil.rmtree(target_path)

            # Перемещаем скачанную подпапку на нужное место
            extracted_path.rename(target_path)
            shutil.rmtree(tmp_path)

            logger.info("Модель успешно извлечена из HF Hub в: %s", target_path)
            return target_path

        except Exception as e:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            logger.error("Сбой скачивания из HF Hub. Временные директории очищены.")
            raise e
