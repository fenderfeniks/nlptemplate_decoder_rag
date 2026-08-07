# src/utils/mlflow/adapter_loader.py
"""Загрузка PEFT LoRA-адаптеров из MLflow artifacts для resume и inference."""

import logging
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def _ensure_tracking_uri() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        logger.warning("MLFLOW_TRACKING_URI не задан. Используется ./mlruns")


def _find_adapter_config(root: Path) -> Path | None:
    """Ищет adapter_config.json рекурсивно — на случай вложенных структур артефактов."""
    for candidate in [root, root / "peft", root / "lora_weights", root / "adapter"]:
        if (candidate / "adapter_config.json").exists():
            return candidate
    matches = list(root.rglob("adapter_config.json"))
    if matches:
        return matches[0].parent
    return None


def _download_by_run_id(run_id: str, artifact_path: str) -> Path:
    logger.info("Скачивание LoRA адаптера по run_id=%s artifact_path=%s", run_id, artifact_path)
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path))


def _download_by_registry(model_name: str, alias: str, artifact_path: str) -> Path:
    logger.info("Поиск LoRA в Registry: %s (alias='%s')", model_name, alias)
    client = MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as e:
        raise MlflowException(
            f"Алиас '{alias}' не найден для модели '{model_name}': {e.message}"
        ) from e

    logger.info(
        "Найдена версия %s (run_id=%s, source=%s), скачиваем...",
        model_version.version,
        model_version.run_id,
        model_version.source,
    )

    if model_version.run_id:
        return Path(
            mlflow.artifacts.download_artifacts(
                run_id=model_version.run_id, artifact_path=artifact_path
            )
        )
    if model_version.source:
        return Path(mlflow.artifacts.download_artifacts(artifact_uri=model_version.source))

    raise MlflowException(
        f"Версия {model_version.version} модели '{model_name}' "
        f"не содержит ни run_id, ни source URI."
    )


def resolve_lora_resume_path(
    resume_cfg: DictConfig | dict,
    tracking_uri: str | None = None,
) -> str | None:
    """Разрешает путь к PEFT-адаптеру из MLflow artifacts.

    Args:
        resume_cfg: Конфиг resume — ожидает ключи ``enabled``, ``run_id``
            или ``model_name`` + ``alias``, ``artifact_path``.
        tracking_uri: MLflow tracking URI. Если None — берётся из
            переменной окружения MLFLOW_TRACKING_URI.

    Returns:
        Локальный путь к директории с adapter_config.json, или None
        если resume отключён (``enabled=false``).
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
    else:
        _ensure_tracking_uri()

    if isinstance(resume_cfg, DictConfig):
        resume_cfg = OmegaConf.to_container(resume_cfg, resolve=True)

    if not resume_cfg or not resume_cfg.get("enabled", False):
        return None

    run_id = resume_cfg.get("run_id")
    model_name = resume_cfg.get("model_name")
    alias = resume_cfg.get("alias")
    artifact_path = resume_cfg.get("artifact_path", "lora_weights")

    if run_id:
        downloaded = _download_by_run_id(run_id, artifact_path)
    elif model_name and alias:
        downloaded = _download_by_registry(model_name, alias, artifact_path)
    else:
        raise ValueError("Укажите 'run_id' или комбинацию 'model_name' + 'alias' для резьюма.")

    adapter_dir = _find_adapter_config(downloaded)
    if adapter_dir is None:
        raise FileNotFoundError(
            f"adapter_config.json не найден в скачанных артефактах: {downloaded}\n"
            f"Содержимое: {list(downloaded.rglob('*'))}"
        )

    logger.info("PEFT-адаптер найден: %s", adapter_dir)
    return str(adapter_dir)
