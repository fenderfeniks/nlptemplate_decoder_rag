# src/tools/merge_lora.py
import gc
import logging
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging
from src.utils.mlflow import resolve_lora_resume_path


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def merge_and_export(cfg: DictConfig) -> None:
    """Сливает LoRA адаптер с базовой моделью и сохраняет монолитную модель на диск."""
    cfg = setup_config(cfg)

    pipeline_cfg = getattr(cfg, cfg.pipeline_name)

    # ── 1. Токенизатор ────────────────────────────────────────────────────
    logger.info("Сборка токенизатора через Hydra...")
    tokenizer = hydra.utils.instantiate(pipeline_cfg.model.tokenizer).build()

    # ── 2. Базовая модель ─────────────────────────────────────────────────
    logger.info("Сборка базовой модели через Hydra-билдер...")
    builder = hydra.utils.instantiate(pipeline_cfg.model.builder)
    builder.lora_resume_path = None  # LoRA навешивается вручную ниже
    base_model = builder.build(tokenizer=tokenizer)

    tracking_uri = cfg.logger.pylightning.tracking_uri
    logger.info("MLflow tracking URI: %s", tracking_uri)

    mlflow_model_name = pipeline_cfg.model.architecture.mlflow_model_name
    lora_cfg = OmegaConf.create(
        {
            "enabled": True,
            "model_name": f"{mlflow_model_name}_LoRA",
            "alias": "Staging",
            "artifact_path": cfg.logger.registry.artifact_path,
        }
    )

    logger.info("Поиск адаптера '%s' (алиас: %s)...", lora_cfg.model_name, lora_cfg.alias)
    lora_path = resolve_lora_resume_path(lora_cfg, tracking_uri=tracking_uri)

    # ── 3. Навешивание и слияние ──────────────────────────────────────────
    logger.info("Навешивание LoRA адаптера...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Слияние весов (Merge and Unload)...")
    merged_model = model.merge_and_unload()

    # Восстанавливаем pad_token_id, если он сбился при слиянии
    if hasattr(merged_model, "generation_config") and getattr(
        merged_model.generation_config, "pad_token_id", None
    ) in (None, -1):
        merged_model.generation_config.pad_token_id = (
            tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # ── 4. Сохранение ─────────────────────────────────────────────────────
    base_model_name = pipeline_cfg.model.builder.model_name_or_path
    model_short_name = Path(base_model_name).name
    output_path = Path(cfg.paths.model_dir) / f"merged_{model_short_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Сохранение монолитной модели в: %s", output_path)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # ── 5. Очистка памяти ─────────────────────────────────────────────────
    del model, merged_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        # empty_cache() возвращает память в пул CUDA, но не обратно ОС.
        # Полное освобождение происходит при завершении процесса.
        torch.cuda.empty_cache()
        logger.debug(
            "CUDA memory после очистки: allocated=%.1f MB, reserved=%.1f MB",
            torch.cuda.memory_allocated() / 1e6,
            torch.cuda.memory_reserved() / 1e6,
        )

    logger.info("Слияние успешно завершено!")


if __name__ == "__main__":
    merge_and_export()
