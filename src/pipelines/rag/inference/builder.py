# src/pipelines/rag/inference/builder.py
"""Сборка inference-энкодера RAG."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def build_inference_encoder(cfg: DictConfig, lora_path: str | None = None) -> tuple:
    """Собрать энкодер (model, pooler, tokenizer) для инференса/оценки/индексации."""

    # 1. Токенизатор
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # 2. Подготовка конфига билдера (отключаем режим обучения)
    builder_cfg = cfg.model.builder.copy()

    if "modifiers" in builder_cfg and builder_cfg.modifiers:
        for mod_key, mod_cfg in builder_cfg.modifiers.items():
            target = mod_cfg.get("_target_", "")

            if "PEFTModifier" in target:
                if lora_path is None:
                    # Чистая базовая модель — отключаем PEFT
                    OmegaConf.update(builder_cfg.modifiers, mod_key, None)
                    logger.info("lora_path=None -> PEFTModifier отключён.")
                else:
                    # Замораживаем веса адаптера для инференса
                    OmegaConf.update(
                        builder_cfg.modifiers[mod_key], "is_trainable", False, force_add=True
                    )
                    OmegaConf.update(
                        builder_cfg.modifiers[mod_key],
                        "gradient_checkpointing",
                        False,
                        force_add=True,
                    )

            elif "FullFinetuningModifier" in target:
                # FullFinetuning не нужен на инференсе
                OmegaConf.update(builder_cfg.modifiers, mod_key, None)

    # 3. Сборка базовой модели через HFModelBuilder
    builder = hydra.utils.instantiate(builder_cfg)
    builder.lora_resume_path = lora_path
    base_model = builder.build(tokenizer=tokenizer)

    # Обязательно переводим в eval()
    base_model.eval()

    # 4. Сборка Pooler
    pooler = hydra.utils.instantiate(cfg.model.pooling)
    pooler.eval()

    return base_model, pooler, tokenizer
