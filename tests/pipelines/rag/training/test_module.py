from unittest.mock import MagicMock

import pytest
import torch

from src.pipelines.rag.training.module import RAGLightningModule


@pytest.fixture
def dummy_components():
    """Фикстура с моками энкодера, пулера и лосса."""
    model = MagicMock()
    # Эмулируем last_hidden_state
    model.return_value.last_hidden_state = torch.ones(2, 3, 4)

    pooler = MagicMock(return_value=torch.ones(2, 4))
    loss_fn = MagicMock(return_value=torch.tensor(1.5))

    return model, pooler, loss_fn


class TestRAGLightningModule:
    def test_forward_pass(self, dummy_components):
        """Проверка forward: проход через модель и пулер."""
        model, pooler, loss_fn = dummy_components
        module = RAGLightningModule(model, pooler, loss_fn, optimizer_cfg=None)

        ids = torch.tensor([[1]])
        mask = torch.tensor([[1]])
        res = module(ids, mask)

        assert res.shape == (2, 4)
        model.assert_called_once_with(input_ids=ids, attention_mask=mask)
        pooler.assert_called_once_with(model.return_value.last_hidden_state, mask)

    def test_shared_step_routing(self, dummy_components):
        """Проверка роутинга в _shared_step (маска has_negative)."""
        model, pooler, loss_fn = dummy_components
        module = RAGLightningModule(model, pooler, loss_fn, optimizer_cfg=None)

        # 1. Без negatives
        batch_no_neg = {
            "query_input_ids": torch.tensor([[1]]),
            "query_attention_mask": torch.tensor([[1]]),
            "pos_input_ids": torch.tensor([[2]]),
            "pos_attention_mask": torch.tensor([[1]]),
        }
        module._shared_step(batch_no_neg)
        # loss_fn вызывается с (q, p, None)
        loss_fn.assert_called_once_with(pooler.return_value, pooler.return_value, None)

        # 2. С negatives
        loss_fn.reset_mock()
        batch_with_neg = {
            **batch_no_neg,
            "neg_input_ids": torch.tensor([[3]]),
            "neg_attention_mask": torch.tensor([[1]]),
            "has_negative": torch.tensor([True]),
        }
        module._shared_step(batch_with_neg)
        loss_fn.assert_called_once_with(
            pooler.return_value, pooler.return_value, pooler.return_value
        )

    def test_training_step_inf_nan_protection(self, dummy_components):
        """Защита от бесконечного или NaN лосса."""
        model, pooler, loss_fn = dummy_components
        module = RAGLightningModule(model, pooler, loss_fn, optimizer_cfg=None)

        # Минимально валидный батч для прохождения forward-пасса
        valid_batch = {
            "query_input_ids": torch.tensor([[1]]),
            "query_attention_mask": torch.tensor([[1]]),
            "pos_input_ids": torch.tensor([[2]]),
            "pos_attention_mask": torch.tensor([[1]]),
        }

        # Проверка Inf
        loss_fn.return_value = torch.tensor(float("inf"))
        assert module.training_step(valid_batch, batch_idx=0) is None

        # Проверка NaN
        loss_fn.return_value = torch.tensor(float("nan"))
        assert module.training_step(valid_batch, batch_idx=0) is None
