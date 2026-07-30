# src/rag_pipeline/training/module.py
import logging
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class RAGLightningModule(pl.LightningModule):
    """LightningModule для обучения энкодеров контрастивными методами."""

    def __init__(
        self,
        model: torch.nn.Module,
        pooler: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.pooler = pooler
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Сохраняем гиперпараметры, кроме самих нейросетевых модулей
        self.save_hyperparameters(ignore=["model", "pooler", "loss_fn"])

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Оставляет в чекпоинте только веса адаптера при использовании PEFT."""
        from peft import PeftModel
        from peft.utils import get_peft_model_state_dict

        if isinstance(self.model, PeftModel):
            checkpoint["state_dict"] = get_peft_model_state_dict(
                self.model, state_dict=checkpoint["state_dict"]
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Прогоняет токены через энкодер и пулер, возвращая 1 вектор на документ."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Большинство энкодеров возвращают last_hidden_state
        token_embeddings = outputs.last_hidden_state

        embeddings = self.pooler(token_embeddings, attention_mask)
        return embeddings

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Единая логика прогона батча для train и val."""
        # 1. Получаем векторы запросов
        q_emb = self(batch["query_input_ids"], batch["query_attention_mask"])

        # 2. Получаем векторы позитивных документов
        p_emb = self(batch["pos_input_ids"], batch["pos_attention_mask"])

        # 3. Получаем векторы негативных документов (если есть в батче)
        n_emb = None
        if "neg_input_ids" in batch and batch["neg_input_ids"] is not None:
            n_emb = self(batch["neg_input_ids"], batch["neg_attention_mask"])

        # 4. Считаем Loss
        loss = self.loss_fn(q_emb, p_emb, n_emb)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)

        if not torch.isfinite(loss):
            logger.warning("batch_idx=%d: loss=%s — пропускаем батч.", batch_idx, loss.item())
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        # Собираем параметры энкодера и пулера (если он обучаемый)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params += [p for p in self.pooler.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("Нет параметров для обучения! Проверь модификаторы.")

        if callable(self.optimizer_cfg):
            optimizer = self.optimizer_cfg(trainable_params)
        else:
            optimizer = instantiate(self.optimizer_cfg, params=trainable_params)

        if self.scheduler_cfg is None:
            return optimizer

        if callable(self.scheduler_cfg):
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = self.scheduler_cfg(optimizer=optimizer, num_training_steps=total_steps)
        else:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
