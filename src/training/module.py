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
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Прокидываем **kwargs на случай кастомных голов, которым нужны sentiment_labels и т.д.
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def _extract_loss_and_logits(self, outputs, batch):
        """
        Универсальный экстрактор.
        Понимает HuggingFace ModelOutput и обычные Python словари (от MultiTaskBERT).
        """
        # 1. Достаем loss
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        else:
            loss = getattr(outputs, "loss", None)

        # 2. Достаем logits
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)

        # 3. Применяем внешнюю функцию потерь, если loss не был вычислен внутри модели
        if loss is None and logits is not None and "labels" in batch and self.loss_fn:
            loss = self.loss_fn(logits, batch["labels"])

        if loss is None:
            raise ValueError(
                "Model didn't return 'loss' and no external loss_fn was able to compute it. "
                "Check your architecture configuration."
            )

        return loss, logits

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = self._extract_loss_and_logits(outputs, batch)

        if logits is not None and "labels" in batch:
            preds = torch.argmax(logits, dim=1)
            self.train_metrics.update(preds, batch["labels"])
            self.log_dict(
                self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = self._extract_loss_and_logits(outputs, batch)

        if logits is not None and "labels" in batch:
            preds = torch.argmax(logits, dim=1)
            self.val_metrics.update(preds, batch["labels"])
            self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True, logger=True)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = self._extract_loss_and_logits(outputs, batch)

        if logits is not None and "labels" in batch:
            preds = torch.argmax(logits, dim=1)
            self.test_metrics.update(preds, batch["labels"])
            self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True, logger=True)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.model.parameters())
        if self.scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
