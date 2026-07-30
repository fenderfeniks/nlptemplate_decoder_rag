# src/rag_pipeline/training/callbacks.py
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.rag_pipeline.utils.vector_db import FAISSVectorDB


logger = logging.getLogger(__name__)


class RetrievalEvaluationCallback(pl.Callback):
    """Callback для подсчета метрик Retrieval (MRR, Recall@K) в конце эпохи."""

    def __init__(self, top_k: int = 10):
        super().__init__()
        self.top_k = top_k
        self.vector_db: FAISSVectorDB | None = None

    def _extract_embeddings(
        self, pl_module: pl.LightningModule, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Прогоняет валидационный датасет через модель."""
        pl_module.eval()
        query_embs, doc_embs = [], []

        with torch.no_grad():
            for batch in dataloader:
                # Переносим батч на устройство модели
                batch = {
                    k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Векторизуем запросы
                q = pl_module(batch["query_input_ids"], batch["query_attention_mask"])
                query_embs.append(q.cpu().numpy())

                # Векторизуем правильные документы
                d = pl_module(batch["pos_input_ids"], batch["pos_attention_mask"])
                doc_embs.append(d.cpu().numpy())

        query_embs_np = np.concatenate(query_embs, axis=0)
        doc_embs_np = np.concatenate(doc_embs, axis=0)

        # Генерируем "фейковые" метаданные: ID документа
        # В реальной задаче здесь будут URL, авторы и т.д.
        metadata = [{"doc_id": i} for i in range(len(doc_embs_np))]

        return query_embs_np, doc_embs_np, metadata

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        logger.info("Запуск оценки Retrieval (FAISS)...")
        val_dataloader = trainer.datamodule.val_dataloader()

        # Получаем векторы
        query_embs, doc_embs, metadata = self._extract_embeddings(pl_module, val_dataloader)

        # Инициализируем БД
        if self.vector_db is None:
            embedding_dim = query_embs.shape[1]
            self.vector_db = FAISSVectorDB(embedding_dim=embedding_dim)
        else:
            self.vector_db.reset()

        # Индексируем документы
        self.vector_db.insert(doc_embs, metadata)

        # Выполняем поиск по запросам
        search_results = self.vector_db.search(query_embs, top_k=self.top_k)

        # Считаем метрики (предполагаем, что i-й запрос соотвествует i-му документу)
        mrr = 0.0
        recall = 0.0

        for i, res_list in enumerate(search_results):
            hit = False
            for rank, res in enumerate(res_list):
                if res["metadata"]["doc_id"] == i:
                    mrr += 1.0 / (rank + 1)
                    hit = True
                    break
            if hit:
                recall += 1.0

        n_queries = len(query_embs)
        mrr /= n_queries
        recall /= n_queries

        # Логируем метрики
        pl_module.log("val_mrr", mrr, sync_dist=True, prog_bar=True)
        pl_module.log(f"val_recall@{self.top_k}", recall, sync_dist=True, prog_bar=True)
        logger.info("Retrieval Eval -> MRR: %.4f | Recall@%d: %.4f", mrr, self.top_k, recall)
