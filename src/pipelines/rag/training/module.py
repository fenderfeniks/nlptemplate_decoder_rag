# src/pipelines/rag/training/module.py
import logging
from typing import Any

import pytorch_lightning as pl
import torch

from src.pipelines.base.training.module import OptimizerMixin


logger = logging.getLogger(__name__)


class RAGLightningModule(OptimizerMixin, pl.LightningModule):
    """LightningModule для контрастивного обучения энкодеров (RAG retriever).

    MRO: RAGLightningModule → OptimizerMixin → pl.LightningModule.
    ``configure_optimizers`` и ``on_save_checkpoint`` берутся из OptimizerMixin.

    Обёртывает энкодер + пулер в единый forward-pass и делегирует
    расчёт loss выбранной функции (MNRL, Triplet и т.д.).
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

        # nn.Module не сериализуемы как гиперпараметры Lightning
        self.save_hyperparameters(ignore=["model", "pooler", "loss_fn"])

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
            # None → Lightning пропускает шаг оптимизатора без backward()
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
