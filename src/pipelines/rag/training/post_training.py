# src/pipelines/rag/training/evaluate.py
"""Post-training evaluation для RAG-энкодера.

Запускается в finally-блоке train.py после training.fit() —
загружает лучший чекпоинт и прогоняет финальную оценку на бенчмарке (test).
"""

import logging

import pytorch_lightning as pl
import torch

from src.pipelines.rag.training.module import RAGLightningModule
from src.utils.torch_utils import register_safe_globals
# Импорт протокола нужен для type hinting, как в декодере
from src.utils.logging.protocol import ExperimentLogger


logger = logging.getLogger(__name__)


def run_post_training_evaluation(
    trainer: pl.Trainer,
    model_module: RAGLightningModule,
    datamodule: pl.LightningDataModule,
    experiment_logger: ExperimentLogger,
) -> float | None:
    """Запускает финальную оценку на лучшем чекпоинте после обучения.

    Загружает лучшие веса LoRA из checkpoint_callback и запускает
    trainer.test(), который оценивает ретривал на бенчмарке через коллбэк.

    Returns:
        float | None: best_model_score из checkpoint_callback.
    """
    best_ckpt_path = trainer.checkpoint_callback.best_model_path

    if not best_ckpt_path:
        logger.warning("Лучший чекпоинт не найден — тест на текущих весах (last state).")

    register_safe_globals()

    if best_ckpt_path:
        logger.info("Загрузка лучших весов из '%s'...", best_ckpt_path)
        checkpoint = torch.load(
            best_ckpt_path,
            map_location=model_module.device,
            weights_only=False,
        )
        lora_state_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k
        }
        logger.info("LoRA-веса загружены (%d тензоров).", len(lora_state_dict))
        model_module.load_state_dict(lora_state_dict, strict=False)

    # Вызываем trainer.test(), он автоматически прогонит test_step и вызовет 
    # on_test_epoch_end в RetrievalEvaluationCallback с правильным MLflow контекстом
    logger.info("Финальная оценка на тестовом бенчмарке (best model)...")
    trainer.test(model=model_module, datamodule=datamodule)

    # Читаем score прямо из коллбэка чекпоинта
    score = trainer.checkpoint_callback.best_model_score
    return float(score) if score is not None else None