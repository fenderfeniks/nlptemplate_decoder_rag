import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging
from src.utils.mlflow import resolve_lora_resume_path


setup_logging()
logger = logging.getLogger(__name__)


class PromoteError(RuntimeError):
    """Ошибка при попытке продвижения модели в Production."""


def _promote(tracking_uri: str, reg_model_name: str) -> None:
    """Продвигает модель из Staging в Production, если она лучше текущей."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    logger.info("MLFLOW_TRACKING_URI: %s", tracking_uri)
    logger.info("Модель в Registry: %s", reg_model_name)

    try:
        staging_mv = client.get_model_version_by_alias(reg_model_name, "Staging")
    except MlflowException as e:
        raise PromoteError(f"Алиас 'Staging' не найден для модели '{reg_model_name}'.") from e

    staging_version = staging_mv.version
    staging_score_str = staging_mv.tags.get("val_loss")

    if staging_score_str is None:
        raise PromoteError("У Staging модели нет тега 'val_loss'. Невозможно оценить качество.")

    staging_score = float(staging_score_str)

    try:
        current_prod = client.get_model_version_by_alias(reg_model_name, "Production")
        if current_prod.version == staging_version:
            logger.warning("Версия %s уже является Production. Промоут пропущен.", staging_version)
            return

        prod_score_str = current_prod.tags.get("val_loss")
        prod_score = float(prod_score_str) if prod_score_str else float("inf")
        logger.info(
            "Текущий Production: версия %s (val_loss=%.4f)", current_prod.version, prod_score
        )
    except MlflowException:
        logger.info("Production алиаса ещё нет — первый промоут.")
        prod_score = float("inf")

    logger.info("Сравнение: Staging (%.4f) vs Production (%.4f)", staging_score, prod_score)

    if staging_score < prod_score:
        client.set_registered_model_alias(reg_model_name, "Production", staging_version)
        logger.info(
            "УСПЕХ! Версия %s (val_loss=%.4f) стала новой Production моделью.",
            staging_version,
            staging_score,
        )
    else:
        logger.warning(
            "ОТКАЗ: Модель в Staging (%.4f) хуже или равна текущей Production (%.4f). "
            "Промоут отменён.",
            staging_score,
            prod_score,
        )


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    pipeline_cfg = getattr(cfg, cfg.pipeline_name)
    tracking_uri = cfg.logger.pylightning.tracking_uri
    mlflow_model_name = pipeline_cfg.model.architecture.mlflow_model_name
    reg_model_name = f"{mlflow_model_name}_LoRA"

    # 1. Продвигаем модель в MLflow (сравниваем валидационный лосс)
    try:
        _promote(tracking_uri=tracking_uri, reg_model_name=reg_model_name)
    except PromoteError as e:
        logger.error("%s", e)
        sys.exit(1)

    # 2. Инициализируем хранилище
    storage_client = hydra.utils.instantiate(cfg.storage)
    uri_prefix = cfg.storage.uri_prefix.rstrip("/")

    # 3. Достаем Production-адаптер из MLflow
    lora_cfg = OmegaConf.create(
        {
            "enabled": True,
            "model_name": reg_model_name,
            "alias": "Production",
            "artifact_path": cfg.logger.registry.artifact_path,
        }
    )
    lora_local_path = resolve_lora_resume_path(lora_cfg, tracking_uri=tracking_uri)
    if not lora_local_path:
        logger.error("Не удалось скачать Production LoRA адаптер из MLflow.")
        sys.exit(1)

    # 4. Загружаем адаптер в Production Storage (S3 / Local)
    remote_adapter_dir = f"adapters/{mlflow_model_name}_prod"
    logger.info("Загрузка адаптера в хранилище: %s", remote_adapter_dir)
    storage_client.upload(local_dir=lora_local_path, remote_path=remote_adapter_dir)

    # 5. Получаем URI базовой модели из конфигов
    model_name_or_path = pipeline_cfg.model.architecture.model_name_or_path
    base_model_uri = pipeline_cfg.model.architecture.get(
        "base_model_uri",
        f"hf://{model_name_or_path}"
        if not model_name_or_path.startswith("hf://")
        else model_name_or_path,
    )

    # 6. Формируем и загружаем Манифест
    manifest = {
        "load_type": "lora",
        "base_model_uri": base_model_uri,
        "lora_uri": f"{uri_prefix}/{remote_adapter_dir}",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_remote_path = f"manifests/{cfg.pipeline_name}_manifest.json"

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_file = Path(tmp_dir) / f"{cfg.pipeline_name}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        # Storage загружает директории, поэтому мы создаем папку с одним файлом манифеста
        # и загружаем её содержимое в корень папки manifests/
        storage_client.upload(local_dir=tmp_dir, remote_path="manifests")

    logger.info(
        "Манифест обновлен. Инференс будет использовать LoRA загрузку. Путь: %s/%s",
        uri_prefix,
        manifest_remote_path,
    )


if __name__ == "__main__":
    main()
