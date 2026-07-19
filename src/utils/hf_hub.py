# src/utils/hf_hub.py
import logging
import os

from huggingface_hub import HfApi, hf_hub_download


logger = logging.getLogger(__name__)


def download_hf_artifact(
    repo_id: str, filename: str, local_dir: str, token: str | None = None
) -> str:
    """
    Скачивает конкретный файл (например, веса модели) из репозитория Hugging Face.

    Args:
        repo_id: ID репозитория (например, "username/my-spam-model")
        filename: Имя файла в репозитории (например, "best_model.ckpt" или "adapter_model.bin")
        local_dir: Локальная папка, куда сохранить файл
        token: HF Токен (если репозиторий приватный). Если None, ищет в os.environ["HUGGINGFACE_TOKEN"].

    Returns:
        str: Полный локальный путь к скачанному файлу.
    """
    logger.info(f"Подготовка к скачиванию {filename} из репозитория {repo_id}...")
    os.makedirs(local_dir, exist_ok=True)

    auth_token = token or os.getenv("HUGGINGFACE_TOKEN")

    try:
        # hf_hub_download использует умное кэширование:
        # Если файл уже скачан и не изменился на сервере, он не будет качать его заново
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

    Args:
        local_file_path: Полный путь к файлу на твоем диске
        repo_id: ID репозитория (например, "username/my-spam-model")
        filename_in_repo: Как файл будет называться внутри репозитория HF
        token: HF Токен. Если None, ищет в os.environ["HUGGINGFACE_TOKEN"].

    Returns:
        str: Прямая ссылка (URL) на загруженный файл.
    """
    logger.info(f"Подготовка к загрузке {local_file_path} в репозиторий {repo_id}...")

    auth_token = token or os.getenv("HUGGINGFACE_TOKEN")
    if not auth_token:
        raise ValueError("Для загрузки артефактов требуется HUGGINGFACE_TOKEN с правами 'Write'.")

    api = HfApi()

    try:
        # Опционально: создаем приватный репозиторий, если его еще не существует
        api.create_repo(repo_id=repo_id, token=auth_token, private=True, exist_ok=True)

        # Загружаем сам файл
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
