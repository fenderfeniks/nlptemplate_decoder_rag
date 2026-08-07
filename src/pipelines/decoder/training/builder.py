"""Сборка LightningModule декодера для обучения.

По структуре зеркалит src/pipelines/rag/training/builder.py:
одна функция возвращает всё что нужно train.py для запуска Trainer.fit().

Отличия от RAG-билдера:
- lora_resume_path — механизм resume адаптера при обучении (в RAG не нужен,
  там адаптер грузится через манифест только для инференса).
- task_mode (sft / cpt) — RAG всегда contrastive, декодер различает режимы.
- torch.compile — опциональная оптимизация, только для декодера.
- scheduler_cfg — опционален, читается из конфига если секция присутствует.
"""

import logging

import hydra
import torch
from omegaconf import DictConfig

from src.pipelines.decoder.training.module import CausalLMLightningModule
from src.pipelines.decoder.training.task_utils import resolve_task_mode
from src.utils.mlflow import resolve_lora_resume_path


logger = logging.getLogger(__name__)


def build_decoder_module(cfg: DictConfig) -> tuple:
    """Собрать CausalLMLightningModule для обучения.

    Args:
        cfg: Корневой конфиг (после setup_config). Читает cfg.decoder_pipeline.

    Returns:
        (model_module, base_model, tokenizer)

        model_module — готовый CausalLMLightningModule, передаётся в Trainer.fit().
        base_model   — нужен в finally-блоке train.py для isinstance(PeftModel) проверки
                       и log_lora_to_mlflow.
        tokenizer    — нужен в log_lora_to_mlflow.
    """
    pipeline_cfg = cfg.decoder_pipeline

    # 1. Токенизатор
    logger.info("Загрузка токенизатора: %s", pipeline_cfg.model.architecture.model_name_or_path)
    tokenizer = hydra.utils.instantiate(pipeline_cfg.model.tokenizer).build()

    # 2. Модель (с опциональным resume LoRA-адаптера)
    lora_resume_path = resolve_lora_resume_path(pipeline_cfg.model.get("lora_resume", {}))

    logger.info("Сборка модели...")
    builder = hydra.utils.instantiate(pipeline_cfg.model.builder)
    builder.lora_resume_path = lora_resume_path
    base_model = builder.build(tokenizer=tokenizer)

    # 3. LightningModule
    model_module = CausalLMLightningModule(
        model=base_model,
        optimizer_cfg=hydra.utils.instantiate(pipeline_cfg.optimizer),
        scheduler_cfg=hydra.utils.instantiate(pipeline_cfg.scheduler)
        if "scheduler" in pipeline_cfg
        else None,
        task_mode=resolve_task_mode(pipeline_cfg.data),
    )

    # 4. torch.compile (опционально)
    if pipeline_cfg.model.get("compile", False):
        logger.info("torch.compile включён — компиляция графа вычислений...")
        model_module.model = torch.compile(model_module.model)

    return model_module, base_model, tokenizer
