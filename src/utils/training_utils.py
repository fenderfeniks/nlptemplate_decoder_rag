"""Утилиты общего назначения для train-скриптов.

Не привязаны к конкретному пайплайну — используются и RAG, и decoder.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def resolve_resume_path(cfg: DictConfig) -> str | None:
    """Возвращает путь к last.ckpt для auto-resume или None.

    Логика:
        - Если ``cfg.resume_training`` не задан или False — возвращает None.
        - Если задан True, но last.ckpt не существует — предупреждает и
          возвращает None (старт с нуля, не падаем).

    Args:
        cfg: Корневой конфиг Hydra. Ожидает ключи:
            ``resume_training`` (bool) и ``paths.log_dir`` (str).

    Returns:
        Абсолютный путь к last.ckpt как строка, или None.
    """
    if not cfg.get("resume_training", False):
        return None

    last_ckpt = Path(cfg.paths.log_dir) / "checkpoints" / "last.ckpt"
    if last_ckpt.exists():
        logger.info("Resume: найден чекпоинт '%s'.", last_ckpt)
        return str(last_ckpt)

    logger.warning("resume_training=True, но last.ckpt не найден — старт с нуля.")
    return None
