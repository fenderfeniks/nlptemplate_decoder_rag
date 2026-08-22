# src/pipelines/decoder/training/module.py
import logging
from typing import Any

import pytorch_lightning as pl
import torch

from src.pipelines.base.training.module import OptimizerMixin


logger = logging.getLogger(__name__)


class CausalLMLightningModule(OptimizerMixin, pl.LightningModule):
    """LightningModule для обучения Causal LM (SFT)."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.save_hyperparameters(ignore=["model"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
        outputs = self(**batch)
        loss = outputs.loss

        if loss is None:
            raise ValueError("Модель не вернула loss. Проверь передачу labels из коллатора.")

        if not torch.isfinite(loss):
            logger.warning("batch_idx=%d: loss=%s — пропускаем батч.", batch_idx, loss.item())
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self(**batch).loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self(**batch).loss
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
