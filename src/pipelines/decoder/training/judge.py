# src/pipelines/decoder/training/judge.py
"""Judge-оценка генераций для GenerationEvaluationCallback.

Вынесено из callbacks.py: judge имеет независимый жизненный цикл
(ленивая инициализация, отдельный cfg) и не должен смешиваться
с логикой генерации и подсчёта метрик.
"""

import logging
from typing import Any

import pandas as pd
import pytorch_lightning as pl


logger = logging.getLogger(__name__)

_MODE_SFT = "sft"


class GenerationJudge:
    """Обёртка над judge-инстансом с ленивой инициализацией и логированием в MLflow.

    Принимает уже готовые prompts/targets/generated из callback —
    не делает повторную генерацию.
    """

    def __init__(
        self,
        judge_cfg: Any | None = None,
        judge_every_n_steps: int | None = None,
    ) -> None:
        self.judge_cfg = judge_cfg
        self.judge_every_n_steps = judge_every_n_steps

        self._judge: Any | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Ленивая инициализация
    # ------------------------------------------------------------------

    def _get_judge(self) -> Any | None:
        """Инстанциирует judge из cfg при первом вызове.

        Ленивая инициализация намеренна:
        - NLI-модель не занимает VRAM во время обучения.
        - LLM-judge не делает лишних сетевых вызовов при старте.
        - Ошибка конфига/сети проявляется только при реальном использовании.
        """
        if self._initialized:
            return self._judge

        self._initialized = True

        if self.judge_cfg is None:
            logger.info("GenerationJudge: judge не задан, оценка отключена.")
            return None

        try:
            from hydra.utils import instantiate

            self._judge = instantiate(self.judge_cfg)
            logger.info(
                "GenerationJudge: инициализирован (%s).",
                self.judge_cfg.get("_target_", "unknown"),
            )
        except Exception as e:
            logger.error("GenerationJudge: сбой инициализации — %s. Оценка будет пропущена.", e)
            self._judge = None

        return self._judge

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def should_run_on_val(self, trainer: pl.Trainer, resolved_mode: str) -> bool:
        """True если judge нужно запустить на текущем валидационном шаге."""
        if self.judge_every_n_steps is None:
            return False
        if resolved_mode != _MODE_SFT:
            return False
        return trainer.global_step > 0 and trainer.global_step % self.judge_every_n_steps == 0

    def maybe_run(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prompts: list[str],
        targets: list[str],
        generated: list[str],
        stage: str,
        resolved_mode: str,
    ) -> None:
        """Запускает judge если условия выполнены, иначе no-op."""
        run = stage == "test" or self.should_run_on_val(trainer, resolved_mode)
        if not run:
            return

        judge = self._get_judge()
        if judge is None:
            return

        self._evaluate_and_log(trainer, pl_module, judge, prompts, targets, generated, stage)

    # ------------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------------

    def _evaluate_and_log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        judge: Any,
        prompts: list[str],
        targets: list[str],
        generated: list[str],
        stage: str,
    ) -> None:
        from src.tools.evaluation.schema import EvalInput

        inputs = [
            EvalInput(prompt=p, response=g, reference=t)
            for p, g, t in zip(prompts, generated, targets)  # noqa: B905
        ]

        try:
            results = judge.evaluate_batch(inputs)
        except Exception as e:
            logger.error("GenerationJudge: evaluate_batch сбой — %s", e)
            return

        scores = [r.score for r in results if r.score is not None]
        verdicts = [r.verdict for r in results if r.verdict is not None]

        if scores:
            avg_score = sum(scores) / len(scores)
            pl_module.log(f"{stage}_judge_score", avg_score, sync_dist=True, prog_bar=True)
            logger.info(
                "GenerationJudge: avg_score=%.4f (n=%d, stage=%s)",
                avg_score,
                len(scores),
                stage,
            )

        if verdicts:
            pl_module.log(
                f"{stage}_judge_pass_rate",
                sum(verdicts) / len(verdicts),
                sync_dist=True,
            )

        self._log_mlflow_table(
            trainer,
            pd.DataFrame(
                {
                    "Prompt": prompts,
                    "Target": targets,
                    "Generated": generated,
                    "Judge Score": [r.score for r in results],
                    "Judge Verdict": [r.verdict for r in results],
                    "Judge Reasoning": [r.reasoning for r in results],
                }
            ),
            stage=stage,
            artifact_suffix="_judge",
            global_step=trainer.global_step,
        )

    @staticmethod
    def _log_mlflow_table(
        trainer: pl.Trainer,
        df: pd.DataFrame,
        stage: str,
        artifact_suffix: str = "",
        global_step: int = 0,
    ) -> None:
        if not (trainer.logger and hasattr(trainer.logger, "experiment")):
            return
        trainer.logger.experiment.log_table(
            run_id=trainer.logger.run_id,
            data=df,
            artifact_file=(f"generations/{stage}_step_{global_step}_results{artifact_suffix}.json"),
        )
