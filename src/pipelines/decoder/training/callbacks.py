# src/pipelines/decoder/training/callbacks.py
import contextlib
import logging
from typing import Any

import pytorch_lightning as pl
import torch

from src.evaluation.evaluators.decoder import DecoderEvaluator


logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Унифицированный доступ к полю конфига — dict или DictConfig/object."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class GenerationEvaluationCallback(pl.Callback):
    """Callback для периодической генерации текста и подсчёта метрик (SFT).

    На валидации и тесте берёт сырые тексты из DataModule:
        val  — из datamodule.val_dataset_raw  (колонки из data_cfg)
        test — из datamodule.test_dataset_raw (колонки prompt/response)

    Генерирует ответы, считает метрики, логирует таблицу через LightningMLflowLogger.
    """

    def __init__(
        self,
        model_name: str,
        experiment_logger: Any,  # сохраняется но не используется для instantiate
        num_random: int = 5,
        generation_batch_size: int = 2,
        generation_kwargs: dict[str, Any] | None = None,
        fixed_samples: list[dict[str, Any]] | None = None,
        metrics_cfg: Any | None = None,
    ) -> None:
        self.model_name = model_name
        # experiment_logger хранится на случай если понадобится run_id снаружи,
        # но создавать metrics_logger будем через LightningMLflowLogger напрямую
        self.experiment_logger = experiment_logger
        self.num_random = num_random
        self.generation_batch_size = generation_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.fixed_samples = fixed_samples or []

        self._env_ready: dict[str, bool] = {"val": False, "test": False}

        self._evaluator = DecoderEvaluator(
            model_name=model_name,
            num_random=num_random,
            generation_batch_size=generation_batch_size,
            generation_kwargs=self.generation_kwargs,
            fixed_samples=self.fixed_samples,
            metrics_cfg=metrics_cfg,
        )

    # ------------------------------------------------------------------
    # Инициализация датасета для генерации
    # ------------------------------------------------------------------

    def _setup_eval_env(self, trainer: pl.Trainer, stage: str) -> None:
        """Берёт сырые тексты из DataModule и передаёт в evaluator.

        val:  datamodule.val_dataset_raw  — колонки из data_cfg (instruction/output и т.д.)
        test: datamodule.test_dataset_raw — колонки prompt/response из бенчмарка
        """
        if self._env_ready[stage]:
            return

        dm = trainer.datamodule

        if stage == "val":
            raw_ds = getattr(dm, "val_dataset_raw", None)
            prompt_col = dm.data_cfg.get("prompt_column", "prompt")
            target_col = dm.data_cfg.get("target_column", "response")
        else:
            raw_ds = getattr(dm, "test_dataset_raw", None)
            # Бенчмарк всегда имеет колонки prompt/response
            prompt_col = "prompt"
            target_col = "response"

        if raw_ds is None:
            logger.warning(
                "GenerationEvaluationCallback: %s_dataset_raw недоступен — "
                "генерация на stage='%s' отключена.",
                stage,
                stage,
            )
            return

        if prompt_col not in raw_ds.column_names or target_col not in raw_ds.column_names:
            logger.warning(
                "GenerationEvaluationCallback: колонки '%s'/'%s' не найдены в %s_dataset_raw "
                "(доступны: %s) — генерация отключена.",
                prompt_col,
                target_col,
                stage,
                raw_ds.column_names,
            )
            return

        n = min(self.num_random * 10, len(raw_ds))
        records = [
            {"prompt": raw_ds[i][prompt_col], "response": raw_ds[i][target_col]} for i in range(n)
        ]

        self._evaluator._eval_datasets[stage] = records
        self._evaluator._env_ready[stage] = True
        self._env_ready[stage] = True

        logger.info(
            "GenerationEvaluationCallback: stage='%s' готов, %d записей для генерации.",
            stage,
            len(records),
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._setup_eval_env(trainer, stage="val")

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._setup_eval_env(trainer, stage="test")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if not self._env_ready["val"]:
            logger.debug("val eval_dataset не готов — генерация пропускается.")
            return

        # Достаем run_id через протокол
        run_id = self.experiment_logger.get_run_id(trainer)

        # Оборачиваем логирование в контекст активного run'а
        ctx = self.experiment_logger.reopen_run(run_id) if run_id else contextlib.nullcontext()
        with ctx:
            self._evaluator.evaluate(
                stage="val",
                metrics_logger=self.experiment_logger,
                trainer=trainer,
                pl_module=pl_module,
                global_step=trainer.global_step,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info("GenerationEvaluationCallback: тест (SFT)...")
        if not self._env_ready["test"]:
            logger.warning("test eval_dataset не готов — генерация на тесте пропускается.")
            return

        # Аналогично для теста
        run_id = self.experiment_logger.get_run_id(trainer)

        ctx = self.experiment_logger.reopen_run(run_id) if run_id else contextlib.nullcontext()
        with ctx:
            self._evaluator.evaluate(
                stage="test",
                metrics_logger=self.experiment_logger,
                trainer=trainer,
                pl_module=pl_module,
                global_step=trainer.global_step,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
