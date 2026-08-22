# src/pipelines/decoder/inference/builder.py
import logging

import hydra
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def build_decoder_model(cfg: DictConfig, lora_path: str | None = None) -> tuple:
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    builder_cfg = cfg.model.builder.copy()

    if "modifiers" in builder_cfg and builder_cfg.modifiers:
        for mod_key, mod_cfg in builder_cfg.modifiers.items():
            target = mod_cfg.get("_target_", "")

            if "PEFTModifier" in target:
                if lora_path is None:
                    # Убираем PEFT полностью — оценивается чистая база
                    OmegaConf.update(builder_cfg.modifiers, mod_key, None)
                    logger.info("lora_path=None -> PEFTModifier отключён.")
                else:
                    # Накатываем веса, но замораживаем для инференса
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
                OmegaConf.update(builder_cfg.modifiers, mod_key, None)

    builder = hydra.utils.instantiate(builder_cfg)
    builder.lora_resume_path = lora_path

    model = builder.build(tokenizer=tokenizer)
    model.eval()

    return model, tokenizer
