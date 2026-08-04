import pytest
import torch

from src.pipelines.rag.training.losses import MultipleNegativesRankingLoss, TripletLossWrapper


class TestMultipleNegativesRankingLoss:
    def test_mnrl_without_hard_negatives(self):
        """Проверка MNRL только на in-batch negatives."""
        loss_fn = MultipleNegativesRankingLoss(scale=2.0, symmetric=False)
        q = torch.randn(4, 16)
        p = torch.randn(4, 16)

        loss = loss_fn(q, p)
        assert loss.dim() == 0  # Должен возвращаться скаляр
        assert loss.item() > 0

    def test_mnrl_with_hard_negatives_and_symmetric(self):
        """Проверка MNRL с hard negatives и симметричным расчетом."""
        loss_fn = MultipleNegativesRankingLoss(scale=1.0, symmetric=True)
        q = torch.randn(2, 8)
        p = torch.randn(2, 8)
        n = torch.randn(2, 8)

        loss = loss_fn(q, p, n)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestTripletLossWrapper:
    def test_triplet_loss_success(self):
        """Успешный проход TripletLossWrapper."""
        loss_fn = TripletLossWrapper(margin=1.0)
        q = torch.randn(2, 8)
        p = torch.randn(2, 8)
        n = torch.randn(2, 8)

        loss = loss_fn(q, p, n)
        assert loss.dim() == 0

    def test_triplet_loss_missing_negatives(self):
        """TripletLossWrapper должен падать, если neg_embeddings=None."""
        loss_fn = TripletLossWrapper()
        q = torch.randn(2, 8)
        p = torch.randn(2, 8)

        with pytest.raises(ValueError, match="требует hard negatives"):
            loss_fn(q, p, neg_embeddings=None)
