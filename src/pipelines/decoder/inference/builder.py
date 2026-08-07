"""Сборка inference-модели декодера.

Используется в eval.py и infer.py — везде где нужна готовая модель
без Lightning / loss / optimizer.

Возвращает (base_model, tokenizer) — уже с навешенным LoRA-адаптером
если lora_path не None. Pooler декодеру не нужен — отсюда и разница
с аналогичным билдером RAG-пайплайна.
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def build_decoder_model(
    cfg: DictConfig,
    lora_path: str | None = None,
) -> tuple:
    """Собрать декодер-модель для инференса/оценки.

    Args:
        cfg: Корневой конфиг (после setup_config). Читает cfg.decoder_pipeline.model.
        lora_path: Локальный путь к LoRA-адаптеру. None — чистый base model.

    Returns:
        (base_model, tokenizer)
    """
    tokenizer = hydra.utils.instantiate(cfg.decoder_pipeline.model.tokenizer).build()

    # Отключаем модификаторы — при инференсе не нужны
    OmegaConf.update(cfg, "decoder_pipeline.model.builder.modifiers", None, force_add=True)

    builder = hydra.utils.instantiate(cfg.decoder_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)

    if lora_path:
        from peft import PeftModel

        logger.info("LoRA: загрузка адаптера из '%s'", lora_path)
        base_model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=False)

    return base_model, tokenizer
