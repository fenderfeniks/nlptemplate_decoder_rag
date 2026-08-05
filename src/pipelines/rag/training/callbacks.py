import logging

import numpy as np
import pytorch_lightning as pl
import torch

from src.vector_store.base import BaseVectorStore
from src.vector_store.faiss_store import FAISSVectorStore


logger = logging.getLogger(__name__)


class RetrievalEvaluationCallback(pl.Callback):
    """Callback для оценки качества ретривала в конце каждой валидационной эпохи."""

    def __init__(self, top_k: int = 10) -> None:
        super().__init__()
        self.top_k = top_k
        self.vector_db: BaseVectorStore | None = None

    @torch.inference_mode()
    def _extract_embeddings(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Прогоняет валидационный DataLoader и возвращает эмбеддинги query и pos."""
        was_training = pl_module.training
        pl_module.eval()

        query_embs: list[np.ndarray] = []
        doc_embs: list[np.ndarray] = []

        for batch in dataloader:
            batch = {
                k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # .float() перед .numpy() — numpy не поддерживает bf16/fp16.
            # .cpu() вызываем сразу чтобы не держать результаты на GPU
            # пока считается следующий батч (снижает пиковое потребление VRAM).
            q = pl_module(batch["query_input_ids"], batch["query_attention_mask"])
            query_embs.append(q.cpu().float().numpy())

            d = pl_module(batch["pos_input_ids"], batch["pos_attention_mask"])
            doc_embs.append(d.cpu().float().numpy())

        if was_training:
            pl_module.train()

        # Явная очистка GPU-кэша после прогона всего val датасета.
        # inference_mode() не освобождает кэш автоматически — делаем это руками
        # чтобы не держать фрагментированную VRAM пока строится FAISS-индекс.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            np.concatenate(query_embs, axis=0),
            np.concatenate(doc_embs, axis=0),
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        logger.info("RetrievalEval: запуск оценки (top_k=%d)...", self.top_k)

        val_dataloader = trainer.datamodule.val_dataloader()
        if val_dataloader is None:
            logger.warning("val_dataloader не задан — RetrievalEval пропущен.")
            return

        query_embs, doc_embs = self._extract_embeddings(pl_module, val_dataloader)
        n_queries = len(query_embs)

        # Предупреждение о потенциальном OOM: храним 2×N эмбеддингов в numpy
        # плюс FAISS дублирует doc_embs в _metadata. Для больших val-сетов
        # рассмотрите уменьшение val_size или переход на Qdrant-бэкенд.
        emb_mb = (query_embs.nbytes + doc_embs.nbytes) / 1024**2
        if emb_mb > 512:
            logger.warning(
                "RetrievalEval: эмбеддинги занимают %.0f МБ RAM. "
                "При OOM уменьшите val_size или top_k.",
                emb_mb,
            )
        embedding_dim = doc_embs.shape[1]

        if self.vector_db is None or self.vector_db.embedding_dim != embedding_dim:
            if self.vector_db is not None:
                logger.warning(
                    "RetrievalEval: embedding_dim изменился (%d → %d) — пересоздаём индекс.",
                    self.vector_db.embedding_dim,
                    embedding_dim,
                )
            # Инстанцируем конкретную реализацию FAISSVectorStore
            self.vector_db = FAISSVectorStore(
                embedding_dim=embedding_dim,
                index_type="flat",
                normalize_embeddings=True,
            )
        else:
            self.vector_db.reset()

        metadata = [{"doc_id": i} for i in range(n_queries)]
        self.vector_db.insert(doc_embs, metadata)

        search_results = self.vector_db.search(query_embs, top_k=self.top_k)

        mrr = 0.0
        recall = 0.0
        ndcg = 0.0

        for i, res_list in enumerate(search_results):
            for rank, res in enumerate(res_list):
                if res["metadata"]["doc_id"] == i:
                    mrr += 1.0 / (rank + 1)
                    recall += 1.0
                    ndcg += 1.0 / np.log2(rank + 2)
                    break

        mrr /= n_queries
        recall /= n_queries
        ndcg /= n_queries

        pl_module.log("val_mrr", mrr, sync_dist=True, prog_bar=True)
        pl_module.log("val_recall_10", recall, sync_dist=True, prog_bar=True, logger=True)
        pl_module.log("val_ndcg_10", ndcg, sync_dist=True, prog_bar=True, logger=True)

        logger.info(
            "RetrievalEval → MRR: %.4f | Recall@%d: %.4f | NDCG@%d: %.4f",
            mrr,
            self.top_k,
            recall,
            self.top_k,
            ndcg,
        )
