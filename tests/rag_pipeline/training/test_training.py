# tests/rag_pipeline/training/test_training.py
from unittest.mock import MagicMock

import pytest
import torch

from src.rag_pipeline.training.losses import MultipleNegativesRankingLoss, TripletLossWrapper
from src.rag_pipeline.training.module import RAGLightningModule


class TestMultipleNegativesRankingLoss:
    @pytest.fixture
    def embeddings(self):
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        # Делаем асимметричные данные
        p = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
        n = torch.tensor([[-1.0, 0.0], [0.0, -1.0]], requires_grad=True)
        return q, p, n

    def test_loss_without_negatives(self, embeddings):
        q, p, _ = embeddings
        loss_fn = MultipleNegativesRankingLoss(scale=1.0)
        loss = loss_fn(q, p)
        assert loss.requires_grad

    def test_loss_with_hard_negatives(self, embeddings):
        """MNRL добавляет hard negatives в расчет."""
        q, p, n = embeddings
        loss_fn = MultipleNegativesRankingLoss(scale=1.0)
        loss = loss_fn(q, p, n)

        assert isinstance(loss, torch.Tensor)

    def test_symmetric_loss(self, embeddings):
        q, p, _ = embeddings
        loss_fn_sym = MultipleNegativesRankingLoss(scale=1.0, symmetric=True)
        loss_fn_asym = MultipleNegativesRankingLoss(scale=1.0, symmetric=False)
        assert loss_fn_sym(q, p).item() != loss_fn_asym(q, p).item()


class TestTripletLossWrapper:
    def test_raises_without_negatives(self):
        loss_fn = TripletLossWrapper()
        with pytest.raises(ValueError):
            loss_fn(torch.rand(2, 4), torch.rand(2, 4), None)


class TestRAGLightningModule:
    @pytest.fixture
    def module(self):
        mock_model = MagicMock()
        # Возвращаем Fake last_hidden_state
        mock_model.return_value.last_hidden_state = torch.rand(2, 5, 8)

        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.rand(2, 8)

        mock_loss = MagicMock()
        mock_loss.return_value = torch.tensor(1.5, requires_grad=True)

        return RAGLightningModule(
            model=mock_model, pooler=mock_pooler, loss_fn=mock_loss, optimizer_cfg=MagicMock()
        )

    def test_forward_pass_calls_model_and_pooler(self, module):
        """Forward должен вызвать энкодер, а затем пулер."""
        ids = torch.randint(0, 100, (2, 5))
        mask = torch.ones(2, 5)

        out = module(ids, mask)

        module.model.assert_called_once()
        module.pooler.assert_called_once()
        assert out.shape == (2, 8)

    def test_training_step_returns_loss(self, module):
        """Проверка шага обучения: должен вернуть скалярный loss."""
        module.log = MagicMock()
        batch = {
            "query_input_ids": torch.randint(0, 100, (2, 5)),
            "query_attention_mask": torch.ones(2, 5),
            "pos_input_ids": torch.randint(0, 100, (2, 5)),
            "pos_attention_mask": torch.ones(2, 5),
        }

        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        module.log.assert_called_with(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
