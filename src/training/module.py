# src/training/module.py
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torchmetrics import MetricCollection

# Импортируем готовые метрики
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class NLPModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any = None,
        loss_fn_cfg: Any = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.save_hyperparameters(ignore=["model"])

        self.loss_fn = instantiate(loss_fn_cfg) if loss_fn_cfg else None

        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )

        # ИСПРАВЛЕНИЕ: Поддержка нестандартных выходов (Multitask / seq2seq)
        if hasattr(outputs, "logits"):
            loss = self.loss_fn(outputs.logits, batch["labels"]) if self.loss_fn else outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
            self.train_metrics.update(preds, batch["labels"])
            self.log_dict(
                self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
        else:
            # Fallback для архитектур, которые сами считают итоговый лосс
            # (например, суммируют лоссы для нескольких голов внутри себя)
            loss = outputs.loss
            if loss is None:
                raise ValueError(
                    "Model didn't return 'loss' and 'logits' are not available. Custom logic required."
                )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )

        if hasattr(outputs, "logits"):
            loss = self.loss_fn(outputs.logits, batch["labels"]) if self.loss_fn else outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
            self.val_metrics.update(preds, batch["labels"])
            self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = outputs.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.model.parameters())
        if self.scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
