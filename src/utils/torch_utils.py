import functools
import logging

import pytorch_lightning as pl
import torch


logger = logging.getLogger(__name__)


BASE_SAFE_GLOBALS = (
    functools.partial,
    torch.optim.AdamW,
)


def _collect_omegaconf_globals() -> list:
    """Возвращает список безопасных классов OmegaConf."""
    try:
        from omegaconf import base, dictconfig, listconfig, nodes

        return [
            listconfig.ListConfig,
            dictconfig.DictConfig,
            base.ContainerMetadata,
            base.Metadata,
            nodes.AnyNode,
            nodes.IntegerNode,
            nodes.FloatNode,
            nodes.BooleanNode,
            nodes.StringNode,
            nodes.EnumNode,
        ]
    except ImportError:
        logger.debug("OmegaConf is not installed, skipping registration.")
        return []


def _collect_lr_scheduler_globals() -> list:
    """Возвращает список безопасных классов PyTorch LR-scheduler."""
    try:
        from torch.optim import lr_scheduler

        return [
            lr_scheduler.StepLR,
            lr_scheduler.LambdaLR,
            lr_scheduler.OneCycleLR,
            lr_scheduler.ReduceLROnPlateau,
            lr_scheduler.CosineAnnealingLR,
            lr_scheduler.CosineAnnealingWarmRestarts,
        ]
    except ImportError:
        logger.debug("PyTorch schedulers are unavailable.")
        return []


def _collect_transformers_globals() -> list:
    """Возвращает список безопасных объектов из Transformers."""
    try:
        from transformers import get_cosine_schedule_with_warmup

        return [get_cosine_schedule_with_warmup]
    except ImportError:
        logger.debug("Transformers is not installed, skipping registration.")
        return []


def register_safe_globals() -> None:
    """Регистрирует безопасные объекты для загрузки чекпоинтов PyTorch."""
    collectors = [
        _collect_omegaconf_globals,
        _collect_lr_scheduler_globals,
        _collect_transformers_globals,
    ]

    # Кортеж конвертируется в список, создавая безопасную копию для дальнейшего мутирования
    safe_globals = list(BASE_SAFE_GLOBALS)
    for collect in collectors:
        safe_globals.extend(collect())

    torch.serialization.add_safe_globals(safe_globals)

    logger.debug(
        "Registered %d safe globals for checkpoint loading.",
        len(safe_globals),
    )


def load_best_lora_weights(model_module: pl.LightningModule, best_ckpt_path: str) -> None:
    """Загружает веса LoRA-адаптера из чекпоинта PyTorch Lightning в PeftModel.

    Ожидается, что чекпоинт был сохранен с помощью OptimizerMixin,
    который оставляет в state_dict только ключи адаптера.
    """
    try:
        from peft import set_peft_model_state_dict
    except ImportError as e:
        raise ImportError("Для загрузки LoRA-весов необходима библиотека peft.") from e

    logger.info("Загрузка лучших весов LoRA из %s", best_ckpt_path)

    # Загружаем чекпоинт на CPU, чтобы избежать спайков памяти на GPU
    checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)

    # PyTorch Lightning оборачивает веса в ключ 'state_dict'
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Загружаем веса в PEFT модель
    set_peft_model_state_dict(model_module.model, state_dict)

    logger.info("Веса LoRA из лучшего чекпоинта успешно применены к модели.")
