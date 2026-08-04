# src/pipelines/decoder/training/module.py
import logging
from typing import Any

import pytorch_lightning as pl
import torch

from src.pipelines.base.training.module import OptimizerMixin


logger = logging.getLogger(__name__)


class CausalLMLightningModule(OptimizerMixin, pl.LightningModule):
    """LightningModule для обучения Causal LM (CPT / SFT / Chat).

    MRO: CausalLMLightningModule → OptimizerMixin → pl.LightningModule.
    ``configure_optimizers`` и ``on_save_checkpoint`` берутся из OptimizerMixin.

    Поддерживает динамическое сохранение: для LoRA сохраняет только адаптеры,
    для Full Fine-Tuning сохраняет все обучаемые веса.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any | None = None,
        task_mode: str = "cpt",
    ) -> None:
        """
        Args:
            model: Causal LM (HF PreTrainedModel или PeftModel).
            optimizer_cfg: DictConfig или callable для создания оптимизатора.
            scheduler_cfg: DictConfig или callable для планировщика LR (опционально).
            task_mode: Режим задачи — ``'cpt'``, ``'sft'`` или ``'chat'``.
                Влияет на набор логируемых метрик: для CPT дополнительно
                считается perplexity.
        """
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.task_mode = task_mode

        # Маршрутизация метрик: избегаем if/else на каждом шаге
        if self.task_mode == "cpt":
            self._compute_extra_metrics = self._log_perplexity
        else:
            self._compute_extra_metrics = lambda loss, phase: None

        # model и _compute_extra_metrics не сериализуемы как гиперпараметры
        self.save_hyperparameters(ignore=["model", "_compute_extra_metrics"])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Прямой проход через Causal LM.

        Args:
            input_ids: ``[B, L]`` — входные токены.
            attention_mask: ``[B, L]`` — маска реальных токенов.
            labels: ``[B, L]`` — целевые токены для расчёта loss.
                Padding-позиции должны быть замаскированы значением -100
                (стандарт HF — коллатор делает это автоматически).

        Returns:
            ``CausalLMOutputWithPast`` с полем ``loss`` (если переданы ``labels``).
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Шаги обучения
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
        outputs = self(**batch)
        loss = outputs.loss

        if loss is None:
            raise ValueError("Модель не вернула loss. Проверь передачу labels из коллатора.")

        if not torch.isfinite(loss):
            logger.warning(
                "batch_idx=%d: loss=%s — пропускаем батч (возврат None).",
                batch_idx,
                loss.item(),
            )
            # None → Lightning пропускает шаг оптимизатора без backward()
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(**batch)
        loss = outputs.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self._compute_extra_metrics(loss, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(**batch)
        loss = outputs.loss

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self._compute_extra_metrics(loss, "test")

    # ------------------------------------------------------------------
    # Метрики
    # ------------------------------------------------------------------

    def _log_perplexity(self, loss: torch.Tensor, phase: str) -> None:
        try:
            perplexity = torch.exp(loss)
            self.log(f"{phase}_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True)
        except OverflowError:
            self.log(f"{phase}_perplexity", float("inf"), on_epoch=True, prog_bar=True, logger=True)
