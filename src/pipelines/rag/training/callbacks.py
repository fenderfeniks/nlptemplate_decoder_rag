"""Callback для оценки качества ретривера во время обучения."""

import contextlib
import logging
from typing import Any

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from src.evaluation.metrics.retriever import RetrieverMetrics
from src.vector_store.base import BaseVectorStore


logger = logging.getLogger(__name__)


class RetrievalEvaluationCallback(pl.Callback):
    """Callback для оценки качества ретривала в конце валидационной и тестовой эпохи."""

    def __init__(
        self,
        vector_db_cfg: DictConfig,
        experiment_logger: Any = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        log_full_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.experiment_logger = experiment_logger
        self.top_k = top_k
        self.log_full_metrics = log_full_metrics

        self._vector_db_cfg: DictConfig = OmegaConf.create(
            {
                k: v
                for k, v in OmegaConf.to_container(vector_db_cfg, resolve=True).items()
                if k != "loader"
            }
        )
        self.vector_db: BaseVectorStore | None = None

        self.evaluator = RetrieverMetrics(
            retrieval_top_k=top_k,
            rerank_top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def _build_vector_db(self, embedding_dim: int) -> BaseVectorStore:
        return hydra.utils.instantiate(self._vector_db_cfg, embedding_dim=embedding_dim)

    @torch.inference_mode()
    def _extract_and_index(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[np.ndarray, int]:
        was_training = pl_module.training
        pl_module.eval()

        query_embs: list[np.ndarray] = []
        n_queries = 0
        is_first_batch = True

        for batch in dataloader:
            batch = {
                k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            q = pl_module(batch["query_input_ids"], batch["query_attention_mask"])
            query_embs.append(q.cpu().float().numpy())

            d = pl_module(batch["pos_input_ids"], batch["pos_attention_mask"])
            d_np = d.cpu().float().numpy()

            if is_first_batch:
                embedding_dim = d_np.shape[1]
                self._prepare_vector_db(embedding_dim)
                is_first_batch = False

            batch_size = d_np.shape[0]
            metadata = [{"doc_id": n_queries + i} for i in range(batch_size)]
            self.vector_db.insert(d_np, metadata)

            n_queries += batch_size

        if was_training:
            pl_module.train()

        return np.concatenate(query_embs, axis=0), n_queries

    def _prepare_vector_db(self, embedding_dim: int) -> None:
        if self.vector_db is None:
            self.vector_db = self._build_vector_db(embedding_dim)
        elif self.vector_db.embedding_dim != embedding_dim:
            self.vector_db = self._build_vector_db(embedding_dim)
        else:
            self.vector_db.reset()

    def _evaluate_stage(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Единая логика для val и test стадий с управлением контекстом логгера."""
        logger.info("RetrievalEval: запуск оценки на stage='%s' (top_k=%d)...", stage, self.top_k)

        dataloader = (
            trainer.datamodule.val_dataloader()
            if stage == "val"
            else trainer.datamodule.test_dataloader()
        )

        if dataloader is None:
            logger.warning("%s_dataloader не задан — RetrievalEval пропущен.", stage)
            return

        query_embs, n_queries = self._extract_and_index(pl_module, dataloader)

        emb_mb = query_embs.nbytes / 1024**2
        if emb_mb > 256:
            logger.warning(
                "RetrievalEval: query эмбеддинги занимают %.0f МБ RAM. При OOM уменьшите %s_size.",
                emb_mb,
                stage,
            )

        search_results = self.vector_db.search(query_embs, top_k=self.top_k)
        ground_truth: list[list[int]] = [[i] for i in range(n_queries)]
        metrics = self.evaluator.compute(search_results, ground_truth)

        # === Выравнивание с Decoder: Управление контекстом логгера ===
        run_id = None
        if self.experiment_logger:
            run_id = self.experiment_logger.get_run_id(trainer)

        ctx = self.experiment_logger.reopen_run(run_id) if run_id else contextlib.nullcontext()

        with ctx:
            k = self.top_k
            recall_bi_key = f"recall_{k}_biencoder"
            fnr_bi_key = f"fnr_{k}_biencoder"
            ndcg_key = f"ndcg_{k}"

            if self.log_full_metrics:
                for name, value in metrics.items():
                    pl_module.log(
                        f"{stage}_{name}",
                        value,
                        sync_dist=True,
                        prog_bar=(name == "mrr"),
                        logger=True,
                    )
            else:
                pl_module.log(
                    f"{stage}_mrr", metrics["mrr"], sync_dist=True, prog_bar=True, logger=True
                )
                pl_module.log(
                    f"{stage}_{recall_bi_key}", metrics[recall_bi_key], sync_dist=True, logger=True
                )
                pl_module.log(f"{stage}_{ndcg_key}", metrics[ndcg_key], sync_dist=True, logger=True)

        logger.info(
            "RetrievalEval (%s) -> MRR: %.4f | Recall@%d (bi): %.4f | NDCG@%d: %.4f | FNR: %.4f",
            stage,
            metrics["mrr"],
            k,
            metrics[recall_bi_key],
            k,
            metrics[ndcg_key],
            metrics[fnr_bi_key],
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._evaluate_stage(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._evaluate_stage(trainer, pl_module, stage="test")
