# src/utils/mlflow/runner.py
"""Утилиты для работы с MLflow run в контексте Lightning Trainer."""

import logging

import mlflow
import pytorch_lightning as pl


logger = logging.getLogger(__name__)


def extract_mlflow_run_id(trainer: pl.Trainer) -> str | None:
    """Извлекает MLflow run_id из логгера тренера или активного run.

    Пробует несколько способов — разные версии Lightning-MLflow интеграции
    хранят run_id под разными атрибутами. Если ни один не сработал,
    обращается к mlflow.active_run() как fallback.

    Args:
        trainer: Lightning Trainer с подключённым логгером.

    Returns:
        run_id как строка, или None если MLflow run не найден.
    """
    if not trainer.logger:
        return None

    for attr in ("run_id", "_run_id", "runid"):
        val = getattr(trainer.logger, attr, None)
        if val:
            return val

    try:
        active = mlflow.active_run()
        if active:
            return active.info.run_id
    except Exception:
        pass

    return None
