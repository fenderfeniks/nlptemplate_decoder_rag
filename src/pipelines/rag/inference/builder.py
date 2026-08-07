"""Сборка inference-энкодера RAG.

Используется в eval.py, infer.py и index_db.py — везде где нужен готовый
энкодер без Lightning / loss / optimizer.

Возвращает (base_model, pooler, tokenizer) — уже с навешенным LoRA-адаптером
если lora_path не None.
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def build_inference_encoder(
    cfg: DictConfig,
    lora_path: str | None = None,
) -> tuple:
    """Собрать энкодер для инференса/оценки/индексации.

    Args:
        cfg: Корневой конфиг (после setup_config). Читает cfg.rag_pipeline.model.
        lora_path: Локальный путь к LoRA-адаптеру. None — чистый base model.

    Returns:
        (base_model, pooler, tokenizer)
    """
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()

    # Отключаем модификаторы — при инференсе не нужны
    OmegaConf.update(cfg, "rag_pipeline.model.builder.modifiers", None, force_add=True)

    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)

    if lora_path:
        from peft import PeftModel

        logger.info("LoRA: загрузка адаптера из '%s'", lora_path)
        base_model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=False)

    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    return base_model, pooler, tokenizer
