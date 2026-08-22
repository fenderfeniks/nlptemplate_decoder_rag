# src/pipelines/decoder/training/builder.py
import logging

import hydra
import torch
from omegaconf import DictConfig

from src.pipelines.decoder.training.module import CausalLMLightningModule
from src.utils.logging.protocol import ExperimentLogger


logger = logging.getLogger(__name__)


def build_decoder_module(
    cfg: DictConfig,
    experiment_logger: ExperimentLogger,
) -> tuple:
    """Собирает токенизатор, модель и LightningModule для обучения.

    Args:
        cfg: Корневой конфиг Hydra.
        experiment_logger: Логгер эксперимента — используется для загрузки
            LoRA-адаптера при resume (load_adapter).
    """

    logger.info("Загрузка токенизатора: %s", cfg.model.architecture.model_name_or_path)
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # Загрузка адаптера через логгер — не через прямой импорт утилиты
    lora_resume_path = experiment_logger.load_adapter(cfg.model.get("lora_resume", {}))

    logger.info("Сборка модели...")
    builder = hydra.utils.instantiate(cfg.model.builder)
    builder.lora_resume_path = lora_resume_path
    base_model = builder.build(tokenizer=tokenizer)

    model_module = CausalLMLightningModule(
        model=base_model,
        optimizer_cfg=hydra.utils.instantiate(cfg.training.optimizer),
        scheduler_cfg=(
            hydra.utils.instantiate(cfg.training.scheduler)
            if "scheduler" in cfg.training  # Исправлено здесь
            else None
        ),
    )

    if cfg.model.get("compile", False):
        logger.info("torch.compile включён — компиляция графа вычислений...")
        model_module.model = torch.compile(model_module.model)

    return model_module, base_model, tokenizer
