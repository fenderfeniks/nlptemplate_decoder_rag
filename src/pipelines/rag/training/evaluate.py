# src/pipelines/rag/training/evaluate.py
"""Post-training evaluation для RAG-энкодера.

Запускается в finally-блоке train.py после training.fit() —
загружает лучший чекпоинт, прогоняет validate() и возвращает val_mrr
для последующего MLflow-логирования.

Не используется в eval.py и infer.py: там своя логика оценки
через батчевый инференс по БД из манифеста.
"""

import logging

import pytorch_lightning as pl
import torch

from src.pipelines.rag.training.module import RAGLightningModule
from src.utils.torch_utils import register_safe_globals


logger = logging.getLogger(__name__)


def run_post_training_evaluation(
    trainer: pl.Trainer,
    model_module: RAGLightningModule,
    datamodule: pl.LightningDataModule,
) -> float | None:
    """Прогоняет валидацию на лучшем чекпоинте и возвращает val_mrr.

    RAG не имеет test_step — качество ретривала меряется через
    RetrievalEvaluationCallback, который регистрируется в
    RAGLightningModule.configure_callbacks() и вызывается в
    on_validation_epoch_end. Поэтому запускаем trainer.validate()
    и читаем val_mrr из logged_metrics.

    Если лучший чекпоинт не найден — валидируем на текущих весах
    (last state), не падаем.

    Args:
        trainer: Обученный Trainer с checkpoint_callback.
        model_module: Текущий RAGLightningModule.
        datamodule: DataModule для валидации.

    Returns:
        val_mrr как float, или None если метрика не найдена.
    """
    best_ckpt_path = trainer.checkpoint_callback.best_model_path

    if not best_ckpt_path:
        logger.warning("Лучший чекпоинт не найден — оцениваем на текущих весах (last state).")
        trainer.validate(model=model_module, datamodule=datamodule)
    else:
        _load_best_lora_weights(trainer, model_module, best_ckpt_path)
        logger.info("Оценка на валидационной выборке (best model)...")
        trainer.validate(model=model_module, datamodule=datamodule)

    return _extract_val_mrr(trainer)


def _load_best_lora_weights(
    trainer: pl.Trainer,
    model_module: RAGLightningModule,
    ckpt_path: str,
) -> None:
    """Загружает только LoRA-веса из чекпоинта в model_module.

    Базовая модель остаётся замороженной — загружаем только lora_-тензоры,
    чтобы не трогать замороженные веса претрейна.
    """
    register_safe_globals()
    logger.info("Загрузка лучших весов из '%s'...", ckpt_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location=model_module.device,
        weights_only=False,
    )
    lora_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}
    model_module.load_state_dict(lora_state_dict, strict=False)

    logger.info("LoRA-веса загружены (%d тензоров).", len(lora_state_dict))


def _extract_val_mrr(trainer: pl.Trainer) -> float | None:
    """Читает val_mrr из logged_metrics или fallback в checkpoint_callback.

    val_mrr пишется через pl_module.log("val_mrr", ...) внутри
    RetrievalEvaluationCallback — появляется в logged_metrics после validate().

    Fallback на best_model_score нужен если ModelCheckpoint настроен
    на monitor='val_mrr' — тогда метрика уже есть в колбэке чекпоинта
    даже без явного вызова validate().
    """
    mrr = trainer.logged_metrics.get("val_mrr")
    if mrr is not None:
        score = float(mrr)
        logger.info("Best model val_mrr = %.4f", score)
        return score

    ckpt_score = trainer.checkpoint_callback.best_model_score
    if ckpt_score is not None:
        return float(ckpt_score)

    logger.warning(
        "val_mrr не найден в logged_metrics и checkpoint_callback. "
        "Убедитесь что monitor='val_mrr' задан в конфиге ModelCheckpoint."
    )
    return None
