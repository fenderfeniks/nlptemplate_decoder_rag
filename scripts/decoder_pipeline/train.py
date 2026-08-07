# scripts/decoder/train.py
"""Оркестратор обучения Causal LM."""

import gc
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import PeftModel


load_dotenv()

from src.pipelines.base.core.data.builder import DataModule  # noqa: E402
from src.pipelines.decoder.training.builder import build_decoder_module  # noqa: E402
from src.pipelines.decoder.training.evaluate import (  # noqa: E402
    extract_mlflow_run_id,
    run_post_training_evaluation,
)
from src.utils.hydra_utils import setup_config  # noqa: E402
from src.utils.logger import setup_logging  # noqa: E402
from src.utils.mlflow import log_lora_to_mlflow  # noqa: E402
from src.utils.torch_utils import register_safe_globals  # noqa: E402


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Обучение Causal LM (SFT / CPT).

    Полный цикл:
        1. Сборка модуля (токенизатор → модель → LightningModule).
        2. DataModule.
        3. Trainer из конфига.
        4. Trainer.fit() с auto-resume из last.ckpt.
        5. Post-training evaluation на лучшем чекпоинте.
        6. Сохранение LoRA-адаптера в MLflow (если PEFT).
    """
    cfg = setup_config(cfg)
    logger.info("Старт обучения...")

    if cfg.decoder_pipeline.training.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.decoder_pipeline.training.accelerator='gpu', но CUDA недоступна. "
            "Используй environment=local для запуска на CPU."
        )

    pl.seed_everything(cfg.seed, workers=True)

    # ── 1. Модуль ────────────────────────────────────────────────────────────
    model_module, base_model, tokenizer = build_decoder_module(cfg)

    # ── 2. DataModule ────────────────────────────────────────────────────────
    logger.info("Инициализация DataModule...")
    datamodule = DataModule(data_cfg=cfg.decoder_pipeline.data, tokenizer=tokenizer)

    # ── 3. Trainer ───────────────────────────────────────────────────────────
    logger.info("Инициализация Trainer...")
    trainer = hydra.utils.instantiate(cfg.decoder_pipeline.training)

    # ── 4. Auto-resume ────────────────────────────────────────────────────────
    resume_path = None
    if cfg.get("resume_training", False):
        last_ckpt = Path(cfg.paths.log_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            resume_path = str(last_ckpt)
            logger.info("Resume: найден чекпоинт %s", resume_path)
        else:
            logger.warning("resume_training=True, но last.ckpt не найден — старт с нуля.")

    # ── 5. Обучение ───────────────────────────────────────────────────────────
    register_safe_globals()
    best_score = None

    try:
        trainer.fit(model=model_module, datamodule=datamodule, ckpt_path=resume_path)
        logger.info("Обучение завершено.")
    except KeyboardInterrupt:
        logger.warning("Прервано (Ctrl+C) — переход к сохранению артефактов...")
    except Exception:
        logger.exception("Критическая ошибка во время обучения:")
        raise
    finally:
        mlflow_run_id = extract_mlflow_run_id(trainer)
        logger.info("MLflow run_id: %s", mlflow_run_id)

        if not getattr(trainer, "tested", False):
            best_score = run_post_training_evaluation(trainer, model_module, datamodule)

        logger.info("Очистка памяти GPU...")
        del trainer
        del datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 6. MLflow: сохранение LoRA-адаптера ─────────────────────────────
        is_peft = isinstance(base_model, PeftModel)
        if is_peft and best_score is not None and mlflow_run_id is not None:
            log_lora_to_mlflow(
                cfg=cfg,
                model_module=model_module,
                tokenizer=tokenizer,
                run_id=mlflow_run_id,
                best_score=best_score,
                pipeline_name="decoder_pipeline",
            )
        elif not is_peft:
            logger.info("Full Fine-Tuning — MLflow регистрация адаптеров пропущена.")
        else:
            logger.warning(
                "MLflow регистрация пропущена: best_score=%s, run_id=%s",
                best_score,
                mlflow_run_id,
            )


if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline

    enforce_pipeline("decoder_pipeline")
    train()
