# src/evaluation/evaluators/decoder.py
"""Оркестратор инференса и подсчёта метрик для Causal LM (decoder)."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from src.evaluation.metrics.generator import GeneratorMetricsPipeline
from src.utils.logging.protocol import ExperimentLogger

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class DecoderEvaluator:
    """Оркестратор генерации и метрик для decoder-пайплайна.

    Данные для генерации всегда передаются снаружи через eval_dataset
    (список dict с ключами 'prompt' и 'response') или заранее через
    _setup_eval_env в GenerationEvaluationCallback.

    Не лезет в DataModule напрямую — это ответственность колбэка.
    """

    def __init__(
        self,
        model_name: str,
        num_random: int = 5,
        generation_batch_size: int = 2,
        generation_kwargs: dict[str, Any] | None = None,
        fixed_samples: list[dict[str, Any]] | None = None,
        metrics_cfg: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_random = num_random
        self.generation_batch_size = generation_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.fixed_samples = fixed_samples or []

        self.metrics_pipeline: GeneratorMetricsPipeline | None = (
            instantiate(metrics_cfg) if metrics_cfg else None
        )

        self._env_ready: dict[str, bool] = {"val": False, "test": False}
        self._generator: Any | None = None
        self._eval_datasets: dict[str, list[dict[str, str]]] = {"val": [], "test": []}

    # ------------------------------------------------------------------
    # Генератор
    # ------------------------------------------------------------------

    def _setup_generator(
        self,
        trainer: pl.Trainer | None,
        pl_module: pl.LightningModule | None,
        model: Any,
        tokenizer: Any,
    ) -> None:
        """Инициализация HFTextGenerator (вызывается перед каждой генерацией)."""
        if self._generator is not None:
            return

        from src.pipelines.decoder.inference.generator import HFTextGenerator

        actual_model = model if model is not None else (
            pl_module.model if pl_module is not None else None
        )
        actual_tokenizer = tokenizer if tokenizer is not None else (
            trainer.datamodule.tokenizer if trainer is not None else None
        )

        if actual_model is None or actual_tokenizer is None:
            raise ValueError("Не удалось разрешить model/tokenizer для генератора.")

        self._generator = HFTextGenerator(
            model=actual_model,
            tokenizer=actual_tokenizer,
            generation_kwargs=self.generation_kwargs,
        )

    # ------------------------------------------------------------------
    # Датасет
    # ------------------------------------------------------------------

    def _setup_dataset(
        self,
        stage: str,
        eval_dataset: list[dict[str, str]] | None,
    ) -> None:
        """Регистрирует eval_dataset для stage.

        Данные должны быть переданы явно — либо через eval_dataset,
        либо заранее через _evaluator._eval_datasets[stage] из колбэка.
        Самостоятельно в DataModule не лезет.
        """
        if self._env_ready[stage]:
            return

        if eval_dataset is not None:
            self._eval_datasets[stage] = eval_dataset
            self._env_ready[stage] = True
            return

        raise ValueError(
            f"eval_dataset для stage='{stage}' не передан и не был инициализирован "
            f"через GenerationEvaluationCallback._setup_eval_env. "
            f"Убедитесь что колбэк добавлен в trainer и DataModule содержит "
            f"{'val_dataset_raw' if stage == 'val' else 'test_dataset_raw'}."
        )

    # ------------------------------------------------------------------
    # Генерация с замером латентности
    # ------------------------------------------------------------------

    def _generate_chunks_with_stats(
        self, prompts: list[str]
    ) -> tuple[list[str], list[dict]]:
        """Генерация с замером per-sample латентности и токенов.

        Returns:
            generated — список сгенерированных строк (в том же порядке).
            stats     — per-sample словари:
                          latency_s        — время инференса (сек.)
                          prompt_tokens    — кол-во токенов входного промпта
                          generated_tokens — кол-во сгенерированных токенов
        """
        generated: list[str] = []
        stats: list[dict] = []

        tokenizer = getattr(self._generator, "tokenizer", None)

        def _count_tokens(text: str) -> int:
            if tokenizer is not None:
                try:
                    return len(tokenizer.encode(text, add_special_tokens=False))
                except Exception:
                    pass
            return max(1, len(text.split()))

        for i in range(0, len(prompts), self.generation_batch_size):
            chunk = prompts[i: i + self.generation_batch_size]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t0 = time.perf_counter()
            chunk_generated = self._generator.generate(chunk)
            elapsed = time.perf_counter() - t0

            per_sample_latency = elapsed / len(chunk)

            for prompt_text, gen_text in zip(chunk, chunk_generated):
                stats.append({
                    "latency_s": per_sample_latency,
                    "prompt_tokens": _count_tokens(prompt_text),
                    "generated_tokens": _count_tokens(gen_text),
                })

            generated.extend(chunk_generated)

        return generated, stats

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        stage: str,
        metrics_logger: ExperimentLogger,
        trainer: pl.Trainer | None = None,
        pl_module: pl.LightningModule | None = None,
        model: Any = None,
        tokenizer: Any = None,
        data_cfg: Any = None,
        global_step: int = 0,
        contexts: list[list[str]] | None = None,
        eval_dataset: list[dict[str, str]] | None = None,
    ) -> dict[str, float]:
        """Запускает eval для stage и возвращает словарь метрик.

        Args:
            stage:          'val' или 'test'.
            metrics_logger: Логгер метрик (LightningMLflowLogger или MLflowLogger).
            trainer:        Lightning Trainer (для доступа к tokenizer).
            pl_module:      Lightning Module (для доступа к model).
            model:          Явная модель (если не через pl_module).
            tokenizer:      Явный токенизатор (если не через trainer.datamodule).
            data_cfg:       Не используется — оставлен для обратной совместимости.
            global_step:    Шаг для логирования.
            contexts:       Контексты для RAG-метрик (опционально).
            eval_dataset:   Список {'prompt': ..., 'response': ...} — если передан,
                            перекрывает данные из колбэка.
        """
        self._setup_generator(trainer, pl_module, model, tokenizer)
        self._setup_dataset(stage, eval_dataset)

        dataset = self._eval_datasets[stage]
        actual_num = min(self.num_random, len(dataset))

        random_samples = [
            {"prompt": item["prompt"], "target": item["response"], "type": "Random"}
            for item in random.sample(dataset, actual_num)
        ]
        fixed_samples = (
            [
                {"prompt": s["prompt"], "target": s["target"], "type": "Fixed"}
                for s in self.fixed_samples
            ]
            if stage == "val" and self.fixed_samples
            else []
        )

        eval_batch = fixed_samples + random_samples
        if not eval_batch:
            logger.warning("DecoderEvaluator: eval_batch пустой — пропускаем.")
            return {}

        prompts = [s["prompt"] for s in eval_batch]
        targets = [s["target"] for s in eval_batch]
        sample_types = [s["type"] for s in eval_batch]

        generated, generation_stats = self._generate_chunks_with_stats(prompts)

        computed_metrics: dict[str, float] = {}
        if self.metrics_pipeline is not None:
            computed_metrics = self.metrics_pipeline.compute_all(
                prompts=prompts,
                generated=generated,
                targets=targets,
                contexts=contexts,
                extra={"generation_stats": generation_stats},
            )

        metrics_logger.log_metrics(
            metrics=computed_metrics,
            stage=stage,
            step=global_step,
        )
        metrics_logger.log_table(
            df=pd.DataFrame({
                "Type": sample_types,
                "Prompt": prompts,
                "Target": targets,
                "Generated": generated,
            }),
            stage=stage,
            step=global_step,
        )

        return computed_metrics