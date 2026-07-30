# src/rag_pipeline/training/callbacks.py
import logging

import numpy as np
import pytorch_lightning as pl
import torch

from src.utils.vector_db import FAISSVectorDB  # единый путь, как в indexer.py


logger = logging.getLogger(__name__)


class RetrievalEvaluationCallback(pl.Callback):
    """Callback для оценки качества ретривала в конце каждой валидационной эпохи.

    Вычисляет метрики MRR@K, Recall@K и NDCG@K на валидационном датасете,
    используя временный FAISS-индекс из positive-документов текущего батча.

    Предполагает, что i-й запрос соответствует i-му позитивному документу
    (стандартное соглашение для contrastive-датасетов).
    """

    def __init__(self, top_k: int = 10) -> None:
        """
        Args:
            top_k: Максимальная глубина ранжирования для всех метрик.
        """
        super().__init__()
        self.top_k = top_k
        self.vector_db: FAISSVectorDB | None = None

    @torch.inference_mode()
    def _extract_embeddings(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Прогоняет валидационный DataLoader и возвращает эмбеддинги query и pos.

        Args:
            pl_module: LightningModule с методом ``forward(input_ids, attention_mask)``.
            dataloader: Валидационный DataLoader (ContrastiveDataCollator).

        Returns:
            Кортеж ``(query_embs, doc_embs)`` — np.ndarray формы ``(N, hidden_size)``.
        """
        was_training = pl_module.training
        pl_module.eval()

        query_embs: list[np.ndarray] = []
        doc_embs: list[np.ndarray] = []

        for batch in dataloader:
            batch = {
                k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            q = pl_module(batch["query_input_ids"], batch["query_attention_mask"])
            query_embs.append(q.cpu().numpy())

            d = pl_module(batch["pos_input_ids"], batch["pos_attention_mask"])
            doc_embs.append(d.cpu().numpy())

        # Восстанавливаем режим модели — Lightning иногда вызывает callback
        # до того как сам вернёт модель в train()
        if was_training:
            pl_module.train()

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

        # Инициализируем или сбрасываем временный FAISS-индекс
        if self.vector_db is None:
            self.vector_db = FAISSVectorDB(
                embedding_dim=doc_embs.shape[1],
                index_type="flat",  # точный поиск для eval — важна воспроизводимость
                normalize_embeddings=True,
            )
        else:
            self.vector_db.reset()

        # Каждый pos-документ получает id = его порядковый номер
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
                    # NDCG@K: relevance=1 для единственного релевантного документа
                    # ideal DCG = 1 / log2(2) = 1.0 (если документ на позиции 0)
                    ndcg += 1.0 / np.log2(rank + 2)  # +2: log2(1)=0, начинаем с rank=0
                    break

        mrr /= n_queries
        recall /= n_queries
        ndcg /= n_queries

        pl_module.log("val_mrr", mrr, sync_dist=True, prog_bar=True)
        pl_module.log(f"val_recall@{self.top_k}", recall, sync_dist=True, prog_bar=True)
        pl_module.log(f"val_ndcg@{self.top_k}", ndcg, sync_dist=True, prog_bar=True)

        logger.info(
            "RetrievalEval → MRR: %.4f | Recall@%d: %.4f | NDCG@%d: %.4f",
            mrr,
            self.top_k,
            recall,
            self.top_k,
            ndcg,
            self.top_k,
        )
