"""Универсальный оркестратор индексации (построения векторных БД)."""

import gc
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from src.tools.storage.resolver import ArtifactResolver
from src.utils.logger import setup_logging
from src.utils.torch_utils import register_safe_globals


setup_logging()
logger = logging.getLogger(__name__)


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_universal_index(
    cfg: DictConfig, pipeline_name: str, run_fn: Callable[[DictConfig, ArtifactResolver, Any], None]
) -> None:
    """Обертка индексации с единой инициализацией и очисткой."""
    logger.info("Старт индексации для пайплайна: %s...", pipeline_name)
    register_safe_globals()

    # 1. Единая инициализация роутера и резолвера
    router = hydra.utils.instantiate(cfg.system.storage_router)
    cache_base = Path(cfg.system.paths.model_dir) / f"{pipeline_name}_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    try:
        # 2. Запуск пайплайн-специфичной логики
        run_fn(cfg, resolver, router)
    except Exception:
        logger.exception("КРИТИЧЕСКАЯ ОШИБКА во время индексации:")
        raise
    finally:
        _free_memory()
        logger.info("Индексация завершена. Память очищена.")
