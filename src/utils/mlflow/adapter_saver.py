# src/utils/mlflow/adapter_saver.py
"""Сохранение PEFT LoRA-адаптеров в MLflow artifacts и Model Registry."""

import gc
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.torch_utils import register_safe_globals


logger = logging.getLogger(__name__)


def _patch_peft_config_for_hydra(model: Any) -> None:
    """Конвертирует OmegaConf-объекты в peft_config в нативные Python-типы.

    Hydra может подсунуть DictConfig/ListConfig внутрь peft_config —
    MLflow не умеет их сериализовать, поэтому патчим перед save_pretrained.
    """
    if not hasattr(model, "peft_config"):
        return
    for _, peft_cfg in model.peft_config.items():
        for key, value in vars(peft_cfg).items():
            if isinstance(value, (ListConfig, DictConfig)):
                setattr(peft_cfg, key, OmegaConf.to_container(value, resolve=True))


def _build_reg_model_name(mlflow_model_name: str, adapter_type: str = "LoRA") -> str:
    """Единственное место, где собирается имя модели в Registry."""
    return f"{mlflow_model_name}_{adapter_type}"


def _save_adapter_to_tempdir(model_to_save: Any, tokenizer: Any, tmp_path: Path) -> None:
    """Сохраняет веса адаптера и токенизатор во временную директорию."""
    logger.info("Сохранение PEFT-адаптера во временную директорию: %s", tmp_path)
    model_to_save.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)

    if not (tmp_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"save_pretrained не создал adapter_config.json в {tmp_path}. "
            f"Содержимое: {list(tmp_path.iterdir())}"
        )

    logger.info("Файлы адаптера: %s", [f.name for f in tmp_path.iterdir()])


def _register_model_version(client: MlflowClient, artifact_uri: str, reg_model_name: str) -> str:
    """Fallback-регистрация через mlflow.register_model."""
    mv_version = mlflow.register_model(model_uri=artifact_uri, name=reg_model_name).version
    logger.info("Зарегистрирована '%s' версия %s.", reg_model_name, mv_version)
    return mv_version


def _create_model_version(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
    reg_model_name: str,
) -> str:
    """Регистрирует версию модели в Registry, с fallback на register_model."""
    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        client.create_registered_model(reg_model_name)
    except Exception:
        pass  # Модель уже существует — это нормально

    try:
        mv = client.create_model_version(name=reg_model_name, source=model_uri, run_id=run_id)
        return mv.version
    except Exception as e:
        logger.warning(
            "client.create_model_version завершился с ошибкой (%s). "
            "Пробуем mlflow.register_model...",
            e,
        )
        return _register_model_version(client, model_uri, reg_model_name)


def log_lora_to_mlflow(
    cfg: Any,
    model_module: Any,
    tokenizer: Any,
    run_id: str,
    pipeline_name: str,
    best_score: float | None = None,
) -> None:
    """Сохраняет PEFT LoRA-адаптер в MLflow через save_pretrained + log_artifacts.

    Args:
        cfg: Корневой конфиг Hydra.
        model_module: Lightning-модуль с атрибутом .model.
        tokenizer: Токенизатор для сохранения вместе с адаптером.
        run_id: MLflow run ID активного эксперимента.
        pipeline_name: Имя пайплайна («rag_pipeline» или «decoder_pipeline»).
            Используется для разрешения ``mlflow_model_name`` из конфига.
        best_score: Лучшее значение метрики — логируется как тег версии.
    """
    logger.info("Подготовка к сохранению LoRA-адаптера в MLflow (run_id=%s)...", run_id)

    gc.collect()
    register_safe_globals()

    model_to_save = model_module.model
    client = MlflowClient()

    pipeline_cfg = OmegaConf.select(cfg, pipeline_name)
    if pipeline_cfg is None:
        raise ValueError(
            f"Пайплайн '{pipeline_name}' не найден в конфиге. Доступные ключи: {list(cfg.keys())}"
        )

    mlflow_model_name = pipeline_cfg.model.architecture.mlflow_model_name
    reg_model_name = _build_reg_model_name(mlflow_model_name)

    registry_cfg = cfg.get("logger", {}).get("registry", {})
    artifact_path = registry_cfg.get("artifact_path", "lora_weights")
    register_on_success = registry_cfg.get("register_on_success", True)
    promote_to_staging = registry_cfg.get("promote_to_staging", True)

    logger.info("Registry модель: %s | artifact_path: %s", reg_model_name, artifact_path)

    _patch_peft_config_for_hydra(model_to_save)

    # ── Шаг 1-2: сохраняем локально и логируем в run ─────────────────────────
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _save_adapter_to_tempdir(model_to_save, tokenizer, tmp_path)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(tmp_path), artifact_path=artifact_path)
            logger.info("LoRA адаптер сохранён в run_id=%s artifact_path=%s", run_id, artifact_path)
            if best_score is not None:
                mlflow.log_metric("promotion_candidate_val_loss", best_score)

    if not register_on_success:
        logger.info("Регистрация в Model Registry отключена (register_on_success=false).")
        return

    # ── Шаг 3: регистрация версии ────────────────────────────────────────────
    mv_version = _create_model_version(client, run_id, artifact_path, reg_model_name)
    logger.info("Зарегистрирована '%s' версия %s.", reg_model_name, mv_version)

    # ── Шаг 4: алиасы и теги ────────────────────────────────────────────────
    if promote_to_staging:
        client.set_registered_model_alias(name=reg_model_name, alias="Staging", version=mv_version)
        logger.info("Модель '%s' версии %s помечена алиасом 'Staging'.", reg_model_name, mv_version)

    if best_score is not None:
        client.set_model_version_tag(reg_model_name, mv_version, "val_loss", str(best_score))
