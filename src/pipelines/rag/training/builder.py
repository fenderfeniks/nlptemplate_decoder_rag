# src/pipelines/rag/training/builder.py
"""Фабрика RAGLightningModule для train-сценария.

Инкапсулирует сборку токенизатора, энкодера, пулера и loss —
всё что нужно train.py до вызова Trainer.fit().

eval.py и infer.py не используют этот модуль: у них своя
логика сборки энкодера (inference_mode, загрузка БД из манифеста).
"""

import logging
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from src.pipelines.rag.training.module import RAGLightningModule
from src.utils.mlflow import resolve_lora_resume_path


logger = logging.getLogger(__name__)


def build_rag_module(cfg: DictConfig) -> tuple[RAGLightningModule, Any, Any]:
    """Собирает RAGLightningModule из корневого конфига Hydra.

    Порядок сборки:
        1. Токенизатор
        2. Энкодер (с опциональным LoRA-resume из MLflow)
        3. Пулер и loss
        4. RAGLightningModule

    Args:
        cfg: Корневой конфиг Hydra. Ожидает секции:
            ``rag_pipeline.model``, ``rag_pipeline.loss``,
            ``rag_pipeline.optimizer``, ``rag_pipeline.scheduler`` (опционально),
            ``vector_db``.

    Returns:
        Кортеж (model_module, base_model, tokenizer):
            - model_module: готовый RAGLightningModule для Trainer.
            - base_model: базовая модель до обёртки в LightningModule —
              нужна для isinstance(base_model, PeftModel) в train.py.
            - tokenizer: токенизатор — нужен для MLflow-логирования адаптера.
    """
    # ── Токенизатор ──────────────────────────────────────────────────────────
    logger.info(
        "Загрузка токенизатора: %s",
        cfg.rag_pipeline.model.builder.model_name_or_path,
    )
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()

    # ── Энкодер ──────────────────────────────────────────────────────────────
    lora_resume_path = resolve_lora_resume_path(cfg.rag_pipeline.model.get("lora_resume", {}))
    logger.info("Сборка энкодера...")
    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    builder.lora_resume_path = lora_resume_path
    base_model = builder.build(tokenizer=tokenizer)

    # ── Пулер и loss ─────────────────────────────────────────────────────────
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)
    loss_fn = hydra.utils.instantiate(cfg.rag_pipeline.loss)

    # ── Планировщик (опционально) ────────────────────────────────────────────
    scheduler_cfg = (
        hydra.utils.instantiate(cfg.rag_pipeline.scheduler)
        if "scheduler" in cfg.rag_pipeline
        else None
    )

    # ── LightningModule ───────────────────────────────────────────────────────
    model_module = RAGLightningModule(
        model=base_model,
        pooler=pooler,
        loss_fn=loss_fn,
        optimizer_cfg=hydra.utils.instantiate(cfg.rag_pipeline.optimizer),
        scheduler_cfg=scheduler_cfg,
    )

    if cfg.rag_pipeline.model.get("compile", False):
        logger.info("torch.compile: компиляция графа энкодера...")
        model_module.model = torch.compile(model_module.model)

    return model_module, base_model, tokenizer
