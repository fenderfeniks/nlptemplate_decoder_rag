"""Универсальный оркестратор обучения для всех пайплайнов."""

import gc
import logging
from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict
from peft import PeftModel

from src.pipelines.base.core.data.builder import DataModule
from src.tools.benchmark.loader import BenchmarkLoader
from src.tools.storage.resolver import ArtifactResolver
from src.utils.logger import setup_logging
from src.utils.torch_utils import register_safe_globals
import hydra


setup_logging()
logger = logging.getLogger(__name__)


def run_universal_train(
    cfg: DictConfig, 
    pipeline_name: str, 
    build_module_fn: Callable
) -> None:
    """Универсальный цикл обучения.
    
    Args:
        cfg: Корневой конфиг Hydra.
        pipeline_name: Имя пайплайна (напр., 'rag_pipeline', 'decoder_pipeline').
        build_module_fn: Фабрика сборки LightningModule. 
            Сигнатура: (cfg, experiment_logger) -> (model_module, base_model, tokenizer)
    """
    logger.info("Старт обучения пайплайна: %s...", pipeline_name)

    if cfg.training.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("accelerator='gpu', но CUDA недоступна.")

    pl.seed_everything(cfg.seed, workers=True)
    
    experiment_logger = hydra.utils.instantiate(cfg.system.logger.experiment_logger)
    router = hydra.utils.instantiate(cfg.system.storage_router)

    # ── 0. Резолвинг артефактов ───────────────────────────────
    if cfg.get("model", {}).get("use_manifest", True):
        logger.info("Загрузка базовой модели через манифест...")
        cache_base = Path(cfg.system.paths.model_dir) / f"{pipeline_name}_cache"
        resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)
        resolver.resolve_and_patch(
            cfg, cfg.system.manifest.uri, pipeline_name=pipeline_name, is_training=True
        )

    # ── 1. Сборка Модуля (Специфично для пайплайна) ───────────
    # Сюда прокинется build_rag_module или build_decoder_module
    model_module, base_model, tokenizer = build_module_fn(cfg, experiment_logger)

    # ── 2. DataModule ─────────────────────────────────────────
    logger.info("Инициализация DataModule...")
    benchmark_loader = BenchmarkLoader(
        router=router,
        cache_dir=cfg.system.paths.benchmark_cache_dir,
        manifest_uri=cfg.system.manifest.uri,
        pipeline_name=pipeline_name, 
    )

    datamodule = DataModule(
        data_cfg=cfg.data,
        processed_data_dir=cfg.system.paths.processed_data_dir,
        tokenizer=tokenizer,
        benchmark_loader=benchmark_loader,
    )

    # ── 3. Trainer и Callbacks ────────────────────────────────
    callbacks = []
    if "callbacks" in cfg.training:
        for cb_name, cb_cfg in cfg.training.callbacks.items():
            # Прокидываем логгер только в те коллбэки, которые отвечают за evaluation
            if cb_name in ("retrieval_eval", "generation_eval"):
                callbacks.append(hydra.utils.instantiate(cb_cfg, experiment_logger=experiment_logger))
            else:
                callbacks.append(hydra.utils.instantiate(cb_cfg))

    with open_dict(cfg.training):
        keys_to_remove = [
            "callbacks", "optimizer", "scheduler", "loss", 
            "primary_metric", "primary_metric_mode", "warmup_steps"
        ]
        for key in keys_to_remove:
            cfg.training.pop(key, None)

    trainer = hydra.utils.instantiate(cfg.training, callbacks=callbacks)

    # ── 4. Auto-resume ────────────────────────────────────────
    register_safe_globals()
    resume_path = None
    if cfg.get("resume_training", False):
        last_ckpt = Path(cfg.system.paths.log_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            resume_path = str(last_ckpt)
            logger.info("Resume из %s", resume_path)

    # ── 5. Цикл Обучения ──────────────────────────────────────
    best_score = None
    try:
        trainer.fit(model=model_module, datamodule=datamodule, ckpt_path=resume_path)
        logger.info("Обучение завершено.")
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем (Ctrl+C). Переход к оценке и сохранению...")
    except Exception:
        logger.exception("Критическая ошибка:")
        raise
    finally:
        run_id = experiment_logger.get_run_id(trainer)
        logger.info("Experiment run_id: %s", run_id)

        # ── 6. Финальная Оценка ───────────────────────────────
        if not getattr(trainer, "tested", False):
            best_ckpt_path = trainer.checkpoint_callback.best_model_path
            if best_ckpt_path:
                logger.info("Загрузка лучших LoRA-весов для теста...")
                checkpoint = torch.load(best_ckpt_path, map_location=model_module.device, weights_only=False)
                lora_sd = {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}
                model_module.load_state_dict(lora_sd, strict=False)
            else:
                logger.warning("Лучший чекпоинт не найден — тест на текущих весах.")
            
            logger.info("Запуск trainer.test()...")
            trainer.test(model=model_module, datamodule=datamodule)
            score = trainer.checkpoint_callback.best_model_score
            best_score = float(score) if score is not None else None

        # Очистка памяти
        del trainer
        del datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 7. Сохранение Адаптера ────────────────────────────
        is_peft = isinstance(base_model, PeftModel)
        if is_peft and best_score is not None and run_id is not None:
            experiment_logger.save_adapter(
                cfg=cfg,
                model_module=model_module,
                tokenizer=tokenizer,
                run_id=run_id,
                best_score=best_score,
                pipeline_name=pipeline_name,
            )