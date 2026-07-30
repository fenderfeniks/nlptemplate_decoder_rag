# src/tools/promote.py
import logging
import sys

import hydra
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def _promote(tracking_uri: str, reg_model_name: str) -> None:
    """Оценивает Staging-модель и выполняет Promotion в Production.

    Сравнивает метрику val_loss версии Staging с текущей версией
    Production. Если Staging превосходит Production, обновляет
    алиасы в MLflow Model Registry.

    Args:
        tracking_uri: MLflow tracking URI из конфига.
        reg_model_name: Имя модели в Registry вида '{mlflow_model_name}_LoRA'.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    logger.info("MLFLOW_TRACKING_URI: %s", tracking_uri)
    logger.info("Модель в Registry: %s", reg_model_name)

    # 1. Берём текущий Staging
    try:
        staging_mv = client.get_model_version_by_alias(reg_model_name, "Staging")
    except MlflowException:
        logger.error("Алиас 'Staging' не найден для модели '%s'.", reg_model_name)
        sys.exit(1)

    staging_version = staging_mv.version
    staging_score_str = staging_mv.tags.get("val_loss")

    if staging_score_str is None:
        logger.error("У Staging модели нет тега 'val_loss'. Невозможно оценить качество.")
        sys.exit(1)

    staging_score = float(staging_score_str)

    # 2. Проверяем текущий Production
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
            "Промоут отменен.",
            staging_score,
            prod_score,
        )


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    tracking_uri = cfg.logger.pylightning.tracking_uri
    mlflow_model_name = cfg.decoder_pipeline.model.architecture.mlflow_model_name
    reg_model_name = f"{mlflow_model_name}_LoRA"

    _promote(tracking_uri=tracking_uri, reg_model_name=reg_model_name)


if __name__ == "__main__":
    main()
