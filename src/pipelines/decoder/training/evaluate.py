"""Post-training утилиты для decoder-пайплайна.

Вынесено из train.py чтобы скрипт не раздувался вспомогательными функциями.
По структуре зеркалит src/pipelines/rag/training/evaluate.py.
"""

import logging

import pytorch_lightning as pl
import torch

from src.pipelines.decoder.training.module import CausalLMLightningModule
from src.utils.torch_utils import register_safe_globals


logger = logging.getLogger(__name__)


def extract_mlflow_run_id(trainer: pl.Trainer) -> str | None:
    """Извлечь MLflow run_id из логгера Trainer-а.

    Пробует несколько атрибутов (разные версии lightning-mlflow логгеров),
    затем fallback на mlflow.active_run().
    """
    if not trainer.logger:
        return None

    for attr in ("run_id", "_run_id", "runid"):
        val = getattr(trainer.logger, attr, None)
        if val:
            return val

    try:
        import mlflow

        active = mlflow.active_run()
        if active:
            return active.info.run_id
    except Exception:
        pass

    return None


def run_post_training_evaluation(
    trainer: pl.Trainer,
    model_module: CausalLMLightningModule,
    datamodule: pl.LightningDataModule,
) -> float | None:
    """Запустить тест на best checkpoint и вернуть лучший скор.

    Если best_model_path не найден — тестируем на текущих весах (last state)
    и возвращаем None, чтобы вызывающий код знал что скор недостоверен.
    """
    best_ckpt_path = trainer.checkpoint_callback.best_model_path

    if not best_ckpt_path:
        logger.warning("Лучший чекпоинт не найден. Запускаем тест на текущих весах (last state)...")
        trainer.test(model=model_module, datamodule=datamodule)
        return None

    register_safe_globals()
    logger.info("Загрузка лучших весов из %s...", best_ckpt_path)

    checkpoint = torch.load(best_ckpt_path, map_location=model_module.device, weights_only=False)
    lora_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}
    logger.info(
        "LoRA тензоров найдено: %d. Ключи: %s",
        len(lora_state_dict),
        list(lora_state_dict.keys())[:5],
    )
    model_module.load_state_dict(lora_state_dict, strict=False)

    logger.info("Тестирование на отложенной выборке (best model)...")
    trainer.test(model=model_module, datamodule=datamodule)

    score = trainer.checkpoint_callback.best_model_score
    return float(score) if score is not None else None
