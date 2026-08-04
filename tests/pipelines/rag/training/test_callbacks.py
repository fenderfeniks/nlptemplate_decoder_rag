# tests/pipelines/rag/training/test_callbacks.py
from unittest.mock import MagicMock, patch

import torch

from src.pipelines.rag.training.callbacks import RetrievalEvaluationCallback


class TestRetrievalEvaluationCallback:
    def test_skip_on_sanity_checking(self):
        """Пропуск логики во время sanity check."""
        cb = RetrievalEvaluationCallback()
        trainer = MagicMock(sanity_checking=True)
        cb.on_validation_epoch_end(trainer, MagicMock())

    # Исправлено: патчим конкретный класс FAISSVectorStore, который инстанцируется в коде
    @patch("src.pipelines.rag.training.callbacks.FAISSVectorStore")
    def test_evaluation_flow(self, mock_faiss_cls):
        """Проверка флоу вычисления MRR/Recall с моком FAISS."""
        cb = RetrievalEvaluationCallback(top_k=5)
        trainer = MagicMock(sanity_checking=False)

        trainer.datamodule.val_dataloader.return_value = [
            {
                "query_input_ids": torch.tensor([[1]]),
                "query_attention_mask": torch.tensor([[1]]),
                "pos_input_ids": torch.tensor([[2]]),
                "pos_attention_mask": torch.tensor([[1]]),
            }
        ]

        pl_module = MagicMock()
        pl_module.device = "cpu"
        pl_module.training = False
        pl_module.return_value = torch.ones(1, 16)

        mock_faiss = mock_faiss_cls.return_value
        mock_faiss.embedding_dim = 16
        mock_faiss.search.return_value = [[{"metadata": {"doc_id": 0}}]]

        cb.on_validation_epoch_end(trainer, pl_module)

        mock_faiss.insert.assert_called_once()
        mock_faiss.search.assert_called_once()

        pl_module.log.assert_any_call("val_mrr", 1.0, sync_dist=True, prog_bar=True)
        pl_module.log.assert_any_call(
            "val_recall_10", 1.0, sync_dist=True, prog_bar=True, logger=True
        )
        pl_module.log.assert_any_call(
            "val_ndcg_10", 1.0, sync_dist=True, prog_bar=True, logger=True
        )
