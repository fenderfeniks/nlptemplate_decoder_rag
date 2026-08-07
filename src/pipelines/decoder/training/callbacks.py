# src/pipelines/decoder/training/callbacks.py
import logging
import random
from typing import Any

import evaluate
import pandas as pd
import pytorch_lightning as pl
import torch


logger = logging.getLogger(__name__)

_MODE_AUTO = "auto"
_MODE_CPT = "cpt"
_MODE_SFT = "sft"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Унифицированный доступ к полю конфига — dict или DictConfig/object."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class GenerationEvaluationCallback(pl.Callback):
    """Callback для периодической генерации текста и подсчёта метрик.

    На валидации и тесте:
    - CPT: генерирует продолжения для случайных промптов, логирует avg_gen_length.
    - SFT: генерирует ответы, считает ROUGE и BLEU, логирует таблицу в MLflow.

    Режим определяется автоматически из конфига данных (``task`` или наличие
    ``prompt_column``), либо задаётся явно через ``mode``.

    Judge-оценка (LLM-as-a-Judge или NLI) запускается только в SFT-режиме
    на ``on_test_epoch_end`` — после завершения обучения. Опционально —
    каждые ``judge_every_n_steps`` шагов в ``on_validation_epoch_end``.
    Judge инстанциируется лениво при первом вызове чтобы не держать
    NLI-модель в памяти во время обучения.
    """

    def __init__(
        self,
        model_name: str,
        num_random: int = 5,
        generation_batch_size: int = 2,
        generation_kwargs: dict[str, Any] | None = None,
        fixed_samples: list[dict[str, Any]] | None = None,
        mode: str = _MODE_AUTO,
        # ── Judge ──────────────────────────────────────────────────────
        judge_cfg: Any | None = None,
        judge_every_n_steps: int | None = None,
    ) -> None:
        """
        Args:
            model_name: Имя модели — тегируется в MLflow для трассировки.
            num_random: Число случайных примеров из датасета на каждый eval.
            generation_batch_size: Размер батча при генерации (контролирует VRAM).
            generation_kwargs: kwargs для ``HFTextGenerator.generate``
                (``max_new_tokens``, ``temperature``, ``do_sample`` и т.д.).
            fixed_samples: Фиксированные примеры для val (список dict с ключами
                ``'prompt'`` и ``'target'``). Логируются каждую эпоху для
                отслеживания прогресса на одних и тех же входах.
            mode: ``'auto'`` — определяется из конфига; ``'cpt'`` или ``'sft'`` — явно.
            judge_cfg: DictConfig узла ``cfg.evaluation.judge``. Если ``None`` —
                judge отключён. Инстанциируется лениво при первом вызове.
            judge_every_n_steps: Запускать judge каждые N глобальных шагов
                на валидации. ``None`` — только на тесте (дефолт).
        """
        self.model_name = model_name
        self.num_random = num_random
        self.generation_batch_size = generation_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.fixed_samples = fixed_samples or []
        self.mode = mode

        self._env_ready: dict[str, bool] = {"val": False, "test": False}
        self._resolved_mode: str | None = None
        self.rouge_metric: Any | None = None
        self.bleu_metric: Any | None = None
        self.generator = None
        self.eval_datasets: dict[str, list[dict[str, str]]] = {"val": [], "test": []}

        from src.pipelines.decoder.training.judge import GenerationJudge

        self._judge = GenerationJudge(judge_cfg, judge_every_n_steps)

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Инициализация окружения
    # ------------------------------------------------------------------

    def _resolve_mode(self, data_cfg: Any) -> str:
        """Определяет режим из конфига если ``mode='auto'``."""
        if self.mode != _MODE_AUTO:
            return self.mode

        task_val = _cfg_get(data_cfg, "task")
        if task_val in (_MODE_SFT, _MODE_CPT):
            resolved = task_val
        else:
            resolved = _MODE_SFT if _cfg_get(data_cfg, "prompt_column") else _MODE_CPT

        logger.info("GenerationEvaluationCallback: mode=auto -> resolved=%s", resolved)
        return resolved

    def _setup_eval_env(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Инициализирует генератор и подготавливает датасет для stage (val/test)."""
        if self._env_ready[stage]:
            return

        from src.pipelines.decoder.inference.generator import HFTextGenerator

        if self.generator is None:
            self.generator = HFTextGenerator(
                model=pl_module.model,
                tokenizer=trainer.datamodule.tokenizer,
                generation_kwargs=self.generation_kwargs,
            )

        data_cfg = trainer.datamodule.data_cfg
        if self._resolved_mode is None:
            self._resolved_mode = self._resolve_mode(data_cfg)

        dataset_key = "validation" if stage == "val" else "test"

        if hasattr(trainer.datamodule, "datasets") and dataset_key in trainer.datamodule.datasets:
            raw_data = trainer.datamodule.datasets[dataset_key]
        else:
            from hydra.utils import instantiate

            logger.warning(
                "Сырой датасет '%s' не найден в памяти, загружаем с диска (source)...",
                dataset_key,
            )
            raw_datasets = instantiate(data_cfg.source).load()
            raw_data = raw_datasets.get(dataset_key, raw_datasets["train"])

        n = min(self.num_random * 10, len(raw_data))

        if self._resolved_mode == _MODE_CPT:
            text_col = _cfg_get(data_cfg, "text_column", "text")
            self.eval_datasets[stage] = [
                {"prompt": raw_data[i][text_col][:200], "response": ""} for i in range(n)
            ]
        else:
            prompt_col = _cfg_get(data_cfg, "prompt_column", "prompt")
            target_col = _cfg_get(data_cfg, "target_column", "completion")
            separator = _cfg_get(data_cfg, "separator", "")
            self.eval_datasets[stage] = [
                {
                    # Separator добавляется к промпту — модель обучалась видеть
                    # "source_text + separator" перед целевым текстом
                    "prompt": raw_data[i][prompt_col] + separator,
                    "response": raw_data[i][target_col],
                }
                for i in range(n)
            ]
            if self.rouge_metric is None:
                self.rouge_metric = evaluate.load("rouge")
            if self.bleu_metric is None:
                self.bleu_metric = evaluate.load("sacrebleu")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            mlflow_client.set_tag(run_id, "model_architecture", self.model_name)
            mlflow_client.set_tag(run_id, "task_type", f"causal_lm_{self._resolved_mode}")

        self._env_ready[stage] = True

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._setup_eval_env(trainer, pl_module, stage="val")

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._setup_eval_env(trainer, pl_module, stage="test")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._resolved_mode is None or trainer.sanity_checking:
            return

        logger.info("GenerationEvaluationCallback: валидация (mode=%s)...", self._resolved_mode)
        if self._resolved_mode == _MODE_SFT:
            self._run_sft_eval(trainer, pl_module, stage="val")
        else:
            self._run_cpt_eval(trainer, pl_module, stage="val")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._resolved_mode is None:
            return

        logger.info("GenerationEvaluationCallback: тест (mode=%s)...", self._resolved_mode)
        if self._resolved_mode == _MODE_SFT:
            self._run_sft_eval(trainer, pl_module, stage="test")
        else:
            self._run_cpt_eval(trainer, pl_module, stage="test")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Генерация
    # ------------------------------------------------------------------

    def _generate_chunks(self, prompts: list[str]) -> list[str]:
        """Генерирует текст батчами для контроля VRAM."""
        generated: list[str] = []
        for i in range(0, len(prompts), self.generation_batch_size):
            chunk = prompts[i : i + self.generation_batch_size]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                generated.extend(self.generator.generate(chunk, **self.generation_kwargs))
        return generated

    # ------------------------------------------------------------------
    # Eval runners
    # ------------------------------------------------------------------

    def _run_sft_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        dataset = self.eval_datasets[stage]
        actual_num = min(self.num_random, len(dataset))
        random_samples = [
            {"prompt": item["prompt"], "target": item["response"], "type": "Random"}
            for item in random.sample(dataset, actual_num)
        ]

        # Фиксированные примеры только на валидации — на тесте нет смысла
        fixed_samples = (
            [
                {"prompt": item["prompt"], "target": item["target"], "type": "Fixed"}
                for item in self.fixed_samples
            ]
            if stage == "val" and self.fixed_samples
            else []
        )

        eval_batch = fixed_samples + random_samples
        if not eval_batch:
            return

        prompts = [s["prompt"] for s in eval_batch]
        targets = [s["target"] for s in eval_batch]
        sample_types = [s["type"] for s in eval_batch]

        generated = self._generate_chunks(prompts)

        rouge_results = self.rouge_metric.compute(
            predictions=generated, references=targets, use_stemmer=True
        )
        rouge1 = self._extract_rouge_score(rouge_results["rouge1"])
        rougeL = self._extract_rouge_score(rouge_results["rougeL"])  # noqa: N806

        bleu_results = self.bleu_metric.compute(
            predictions=generated, references=[[t] for t in targets]
        )
        bleu = bleu_results["score"]

        avg_len = sum(len(t.split()) for t in generated) / len(generated)

        pl_module.log(f"{stage}_rouge1", rouge1, sync_dist=True, prog_bar=True)
        pl_module.log(f"{stage}_rougeL", rougeL, sync_dist=True, prog_bar=True)
        pl_module.log(f"{stage}_bleu", bleu, sync_dist=True, prog_bar=True)
        pl_module.log(f"{stage}_avg_gen_length", avg_len, sync_dist=True)

        self._log_mlflow_table(
            trainer,
            pd.DataFrame(
                {
                    "Type": sample_types,
                    "Prompt": prompts,
                    "Target": targets,
                    "Generated": generated,
                }
            ),
            stage,
        )

        # ── Judge — после логирования основных метрик ──────────────────
        # На тесте: всегда. На валидации: только если задан every_n_steps.
        self._judge.maybe_run(
            trainer, pl_module, prompts, targets, generated, stage, self._resolved_mode
        )

    def _run_cpt_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        dataset = self.eval_datasets[stage]
        actual_num = min(self.num_random, len(dataset))
        prompts = [item["prompt"] for item in random.sample(dataset, actual_num)]

        if not prompts:
            return

        generated = self._generate_chunks(prompts)
        avg_len = sum(len(t.split()) for t in generated) / len(generated)
        pl_module.log(f"{stage}_avg_gen_length", avg_len, sync_dist=True, prog_bar=True)

        self._log_mlflow_table(
            trainer,
            pd.DataFrame(
                {
                    "Prompt (first 200 chars)": prompts,
                    "Generated continuation": generated,
                }
            ),
            stage,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_rouge_score(score: Any) -> float:
        if hasattr(score, "mid"):
            return float(score.mid.fmeasure)
        if isinstance(score, (list, tuple)) and score:
            return float(score[0])
        return float(score)

    @staticmethod
    def _log_mlflow_table(
        trainer: pl.Trainer,
        df: pd.DataFrame,
        stage: str,
        artifact_suffix: str = "",
    ) -> None:
        if not (trainer.logger and hasattr(trainer.logger, "experiment")):
            return
        trainer.logger.experiment.log_table(
            run_id=trainer.logger.run_id,
            data=df,
            artifact_file=(
                f"generations/{stage}_step_{trainer.global_step}_results{artifact_suffix}.json"
            ),
        )
