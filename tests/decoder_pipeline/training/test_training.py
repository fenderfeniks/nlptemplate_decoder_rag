# tests/decoder_pipeline/training/test_training.py
from unittest.mock import MagicMock

import pytest
import torch

from src.decoder_pipeline.training.callbacks import GenerationEvaluationCallback
from src.decoder_pipeline.training.module import CausalLMLightningModule


class TestCausalLMLightningModule:
    @pytest.fixture
    def module(self):
        mock_model = MagicMock()
        mock_model.return_value.loss = torch.tensor(2.5, requires_grad=True)
        return CausalLMLightningModule(model=mock_model, optimizer_cfg=MagicMock())

    def test_training_step_returns_loss(self, module):
        module.log = MagicMock()
        # Добавлен attention_mask
        batch = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_training_step_skips_infinite_loss(self, module):
        module.model.return_value.loss = torch.tensor(float("inf"))
        batch = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        loss = module.training_step(batch, batch_idx=0)
        assert loss is None

    def test_perplexity_calculation(self, module):
        """Проверка функции логгирования перплексии."""
        module.log = MagicMock()
        # loss = 2.0 -> ppl = e^2.0 ≈ 7.389
        module._log_perplexity(torch.tensor(2.0), "val")

        call_args = module.log.call_args[0]
        assert call_args[0] == "val_perplexity"
        assert abs(call_args[1].item() - 7.389) < 0.01


class TestGenerationEvaluationCallback:
    def test_resolve_mode_cpt(self):
        cb = GenerationEvaluationCallback(model_name="test", mode="auto")
        data_cfg = {"text_column": "text"}  # Нет prompt_column -> CPT
        assert cb._resolve_mode(data_cfg) == "cpt"

    def test_resolve_mode_sft(self):
        cb = GenerationEvaluationCallback(model_name="test", mode="auto")
        data_cfg = {"prompt_column": "prompt", "target_column": "response"}
        assert cb._resolve_mode(data_cfg) == "sft"
