import logging
import os

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


logger = logging.getLogger(__name__)


def download_hf_artifact(
    repo_id: str, filename: str, local_dir: str, token: str | None = None
) -> str:
    """
    Скачивает файл или директорию из репозитория Hugging Face.
    """
    logger.info(f"Подготовка к скачиванию {filename} из репозитория {repo_id}...")
    os.makedirs(local_dir, exist_ok=True)

    auth_token = token or os.getenv("HUGGINGFACE_TOKEN")

    try:
        # ИСПРАВЛЕНИЕ: Умная проверка на директорию
        if "." not in filename:
            logger.info(f"Распознана директория. Запуск snapshot_download для {filename}...")
            local_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{filename}/*",
                local_dir=local_dir,
                token=auth_token,
            )
            full_path = os.path.join(local_path, filename)
            logger.info(f"Директория успешно скачана: {full_path}")
            return full_path
        else:
            local_file_path = hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=local_dir, token=auth_token
            )
            logger.info(f"Файл успешно скачан/найден в кэше: {local_file_path}")
            return local_file_path

    except Exception as e:
        logger.error(f"Ошибка при скачивании артефакта из Hugging Face: {e}")
        raise


def upload_hf_artifact(
    local_file_path: str, repo_id: str, filename_in_repo: str, token: str | None = None
) -> str:
    """
    Загружает локальный файл (например, чекпоинт весов) в репозиторий Hugging Face.
    """
    logger.info(f"Подготовка к загрузке {local_file_path} в репозиторий {repo_id}...")

    auth_token = token or os.getenv("HUGGINGFACE_TOKEN")
    if not auth_token:
        raise ValueError("Для загрузки артефактов требуется HUGGINGFACE_TOKEN с правами 'Write'.")

    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, token=auth_token, private=True, exist_ok=True)

        file_url = api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=filename_in_repo,
            repo_id=repo_id,
            token=auth_token,
        )
        logger.info(f"Файл успешно загружен на Hugging Face: {file_url}")
        return file_url

    except Exception as e:
        logger.error(f"Ошибка при загрузке артефакта на Hugging Face: {e}")
        raise
