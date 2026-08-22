import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from src.tools.storage.base import BaseStorage


logger = logging.getLogger(__name__)


class HFHubStorage(BaseStorage):
    """Хранилище Hugging Face Hub с защитой от битых кэшей и симлинков."""

    def __init__(
        self, repo_id: str, uri_prefix: str, token: str | None = None, repo_type: str = "model"
    ) -> None:
        super().__init__(uri_prefix=uri_prefix)
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

        # --- ДОБАВЛЕННЫЙ БЛОК ---
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
            if remote_path in files:
                if target_path.suffix == "":
                    target_path = target_path / Path(remote_path).name
                return self.download_file(remote_path, target_path)
        except Exception as e:
            logger.warning(
                "Не удалось проверить файлы в HF Hub, продолжаем скачивание как директории: %s", e
            )
        # ------------------------

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

    def download_file(self, remote_path: str, local_path: Path | str) -> Path:
        target = Path(local_path)
        remote_path = remote_path.strip("/")
        tmp = target.with_name(target.name + ".tmp")

        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Скачивание файла из HF Hub: %s/%s", self.repo_id, remote_path)
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                token=self.token,
                repo_type=self.repo_type,
                local_dir=str(tmp.parent),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download кладёт файл по пути local_dir/filename
            # переименовываем в tmp и затем в target атомарно
            downloaded_path = Path(downloaded)
            downloaded_path.rename(tmp)

            if target.exists():
                target.unlink()
            tmp.rename(target)

            logger.info("Файл атомарно скачан из HF Hub в: %s", target)
            return target

        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            logger.error("Сбой скачивания файла из HF Hub: %s", e)
            raise

    def exists(self, remote_path: str) -> bool:
        """Проверяет наличие файла или директории в репозитории HF Hub."""
        try:
            # Скачиваем плоский список всех путей к файлам в репозитории
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)

            # Если remote_path указывает на конкретный файл
            if remote_path in files:
                return True

            # Если remote_path указывает на директорию (добавляем слэш,
            # чтобы 'models/model_v1' не сматчило 'models/model_v10')
            prefix = remote_path.rstrip("/") + "/"
            return any(f.startswith(prefix) for f in files)

        except RepositoryNotFoundError:
            logger.warning("Репозиторий %s не найден в HF Hub.", self.repo_id)
            return False
        except Exception as e:
            logger.error("Ошибка при проверке пути '%s' в HF Hub: %s", remote_path, e)
            return False
