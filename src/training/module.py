import pytorch_lightning as pl
import torch
from typing import Any
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
        num_classes: int = 2 # Берем из конфига для инициализации метрик
    ):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        
        # Сохраняем гиперпараметры в MLflow! 
        # (Эта строчка автоматически залогирует всё, что пришло в __init__)
        self.save_hyperparameters(ignore=['model'])
        
        # 1. Динамический Loss из конфига (например, наш FocalLoss)
        self.loss_fn = instantiate(loss_fn_cfg) if loss_fn_cfg else None

        # 2. Инициализация индустриальных метрик
        metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=num_classes, average='macro'),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro')
        })
        # Разделяем метрики для трейна и валидации
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        # Прогоняем данные
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels")
        )
        
        # Считаем Loss (Кастомный или встроенный)
        loss = self.loss_fn(outputs.logits, batch["labels"]) if self.loss_fn else outputs.loss

        # Считаем и логируем метрики
        # Lightning + torchmetrics сами всё усреднят по эпохе!
        preds = torch.argmax(outputs.logits, dim=1)
        self.train_metrics.update(preds, batch["labels"])
        
        # Отправляем в MLflow (logger=True делает магию)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels")
        )
        loss = self.loss_fn(outputs.logits, batch["labels"]) if self.loss_fn else outputs.loss
        
        preds = torch.argmax(outputs.logits, dim=1)
        self.val_metrics.update(preds, batch["labels"])
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.model.parameters())
        if self.scheduler_cfg is None:
            return optimizer
            
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }