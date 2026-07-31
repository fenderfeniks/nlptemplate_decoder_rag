# src/rag_pipeline/training/module.py
import logging
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class RAGLightningModule(pl.LightningModule):
    """LightningModule для контрастивного обучения энкодеров (RAG retriever).

    Обёртывает энкодер + пулер в единый forward-pass и делегирует
    расчёт Loss выбранной функции (MNRL, Triplet и т.д.).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pooler: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any | None = None,
    ) -> None:
        """
        Args:
            model: Энкодер (HF PreTrainedModel или PeftModel).
            pooler: Пулер (``Pooler`` из pooling.py).
            loss_fn: Функция потерь (``MultipleNegativesRankingLoss`` и т.д.).
            optimizer_cfg: DictConfig или callable для создания оптимизатора.
            scheduler_cfg: DictConfig или callable для планировщика LR (опционально).
        """
        super().__init__()
        self.model = model
        self.pooler = pooler
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Сохраняем конфиги гиперпараметров, исключая nn.Module — они не сериализуемы
        self.save_hyperparameters(ignore=["model", "pooler", "loss_fn"])

    # ------------------------------------------------------------------
    # Чекпоинтинг
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """При наличии PEFT — сохраняет только веса адаптера, не базовой модели."""
        try:
            from peft import PeftModel
            from peft.utils import get_peft_model_state_dict
        except ImportError:
            return

        if isinstance(self.model, PeftModel):
            checkpoint["state_dict"] = get_peft_model_state_dict(
                self.model, state_dict=checkpoint["state_dict"]
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Прогоняет токены через энкодер и пулер.

        Args:
            input_ids: ``[B, L]`` — токены.
            attention_mask: ``[B, L]`` — маска реальных токенов.

        Returns:
            ``[B, H]`` — нормализованные эмбеддинги документов/запросов.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.pooler(outputs.last_hidden_state, attention_mask)

    # ------------------------------------------------------------------
    # Шаги обучения
    # ------------------------------------------------------------------

    def _shared_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Единая логика прогона батча для train и val.

        Учитывает маску ``has_negative`` из ``ContrastiveDataCollator``:
        для примеров без hard negatives передаёт их эмбеддинги как None,
        чтобы loss-функция корректно работала в смешанных батчах.
        """
        q_emb = self(batch["query_input_ids"], batch["query_attention_mask"])
        p_emb = self(batch["pos_input_ids"], batch["pos_attention_mask"])

        n_emb: torch.Tensor | None = None
        if "neg_input_ids" in batch:
            has_negative: torch.Tensor = batch.get(
                "has_negative", torch.ones(len(q_emb), dtype=torch.bool)
            )
            if has_negative.any():
                # Прогоняем только те примеры, у которых есть негатив
                neg_ids = batch["neg_input_ids"][has_negative]
                neg_mask = batch["neg_attention_mask"][has_negative]
                n_emb = self(neg_ids, neg_mask)

        return self.loss_fn(q_emb, p_emb, n_emb)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor | None:
        loss = self._shared_step(batch)

        if not torch.isfinite(loss):
            logger.warning(
                "batch_idx=%d: loss не конечен (%s) — батч пропущен. "
                "Проверьте scale, норму градиентов и данные.",
                batch_idx,
                loss.item(),
            )
            # Возвращаем None: Lightning пропускает шаг оптимизатора.
            # ВАЖНО: не возвращаем loss.detach() — иначе backward() вызовется
            # на не-конечном значении и обнулит все параметры.
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        loss = self._shared_step(batch)

        if not torch.isfinite(loss):
            logger.warning(
                "val batch_idx=%d: loss не конечен (%s) — шаг пропущен.",
                batch_idx,
                loss.item(),
            )
            return

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    # ------------------------------------------------------------------
    # Оптимизация
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        """Собирает оптимизатор и (опционально) планировщик LR.

        Параметры для обучения берутся из model + pooler с фильтром
        ``requires_grad=True``. Это корректно работает как для FullFT,
        так и для LoRA (где базовые слои заморожены).
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad] + [
            p for p in self.pooler.parameters() if p.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError(
                "Нет обучаемых параметров. Проверьте модификаторы: все параметры заморожены."
            )

        if callable(self.optimizer_cfg):
            optimizer = self.optimizer_cfg(trainable_params)
        else:
            optimizer = instantiate(self.optimizer_cfg, params=trainable_params)

        if self.scheduler_cfg is None:
            return optimizer

        if callable(self.scheduler_cfg):
            total_steps = self.trainer.estimated_stepping_batches
            if total_steps == float("inf"):
                raise ValueError(
                    "estimated_stepping_batches=inf: задайте max_steps в конфиге тренера."
                )
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
