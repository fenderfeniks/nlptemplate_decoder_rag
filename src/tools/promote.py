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


class PromoteError(RuntimeError):
    """Ошибка при попытке продвижения модели в Production."""


def _promote(tracking_uri: str, reg_model_name: str) -> None:
    """Продвигает модель из Staging в Production, если она лучше текущей.

    Args:
        tracking_uri: URI MLflow Tracking Server.
        reg_model_name: Имя зарегистрированной модели в Registry.

    Raises:
        PromoteError: Если алиас 'Staging' не найден или у модели нет тега val_loss.
    """
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

    tracking_uri = cfg.logger.pylightning.tracking_uri
    pipeline_cfg = getattr(cfg, cfg.pipeline_name)
    mlflow_model_name = pipeline_cfg.model.architecture.mlflow_model_name
    reg_model_name = f"{mlflow_model_name}_LoRA"

    try:
        _promote(tracking_uri=tracking_uri, reg_model_name=reg_model_name)
    except PromoteError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
