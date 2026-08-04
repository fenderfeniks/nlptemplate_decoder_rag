# tests/pipelines/decoder/training/test_module.py
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.pipelines.decoder.training.module import CausalLMLightningModule


@pytest.fixture
def dummy_model():
    """Фейковая модель, которая возвращает объект с полем loss."""

    class DummyOutput:
        def __init__(self, loss=None):
            self.loss = loss

    model = MagicMock()
    model.return_value = DummyOutput(loss=torch.tensor(2.5))
    return model


@pytest.fixture
def dummy_batch():
    """Валидный батч, содержащий обязательные аргументы для forward()."""
    return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}


class TestCausalLMLightningModule:
    def test_forward_pass(self, dummy_model, dummy_batch):
        """Проверка, что аргументы корректно пробрасываются в модель."""
        module = CausalLMLightningModule(dummy_model, optimizer_cfg=None)

        output = module(**dummy_batch)

        assert output.loss.item() == 2.5
        dummy_model.assert_called_once_with(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
            labels=None,
        )

    def test_training_step_missing_loss(self, dummy_model, dummy_batch):
        """Ошибка, если модель вернула None вместо лосса."""
        dummy_model.return_value.loss = None
        module = CausalLMLightningModule(dummy_model, optimizer_cfg=None)

        with pytest.raises(ValueError, match="Модель не вернула loss"):
            module.training_step(dummy_batch, 0)

    def test_training_step_skip_invalid_loss(self, dummy_model, dummy_batch):
        """Пропуск батча (возврат None), если loss NaN или Inf."""
        module = CausalLMLightningModule(dummy_model, optimizer_cfg=None)

        # Test NaN
        dummy_model.return_value.loss = torch.tensor(float("nan"))
        assert module.training_step(dummy_batch, 0) is None

        # Test Inf
        dummy_model.return_value.loss = torch.tensor(float("inf"))
        assert module.training_step(dummy_batch, 0) is None

    def test_training_step_success(self, dummy_model, dummy_batch):
        """Успешный шаг обучения с логированием."""
        module = CausalLMLightningModule(dummy_model, optimizer_cfg=None)
        module.log = MagicMock()

        loss = module.training_step(dummy_batch, batch_idx=0)

        assert loss.item() == 2.5
        module.log.assert_called_once_with(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def test_validation_and_test_step_sft(self, dummy_model, dummy_batch):
        """Проверка режимов валидации/теста для SFT (без перплексии)."""
        module = CausalLMLightningModule(dummy_model, optimizer_cfg=None, task_mode="sft")
        module.log = MagicMock()

        module.validation_step(dummy_batch, 0)
        module.log.assert_called_once_with(
            "val_loss", dummy_model.return_value.loss, on_epoch=True, prog_bar=True, logger=True
        )

        module.log.reset_mock()
        module.test_step(dummy_batch, 0)
        module.log.assert_called_once_with(
            "test_loss", dummy_model.return_value.loss, on_epoch=True, prog_bar=True, logger=True
        )

    def test_validation_step_cpt_logs_perplexity(self, dummy_model, dummy_batch):
        """Проверка режима CPT, который должен дополнительно логировать perplexity."""
        module = CausalLMLightningModule(dummy_model, optimizer_cfg=None, task_mode="cpt")
        module.log = MagicMock()

        module.validation_step(dummy_batch, 0)

        # Должно быть два вызова log: один для loss, второй для perplexity (exp(2.5) ~ 12.18)
        assert module.log.call_count == 2
        call_args = module.log.call_args_list

        assert call_args[0].args[0] == "val_loss"
        assert call_args[1].args[0] == "val_perplexity"
        assert abs(call_args[1].args[1].item() - 12.18) < 0.1

    def test_perplexity_overflow_protection(self, dummy_model):
        """Проверка защиты от переполнения при расчете perplexity (очень большой loss)."""
        # Loss 1000 приведет к переполнению в torch.exp
        dummy_model.return_value.loss = torch.tensor(1000.0)
        module = CausalLMLightningModule(dummy_model, optimizer_cfg=None, task_mode="cpt")
        module.log = MagicMock()

        # Мокаем torch.exp через стандартный unittest.mock.patch
        with patch("torch.exp", side_effect=OverflowError):
            module._log_perplexity(dummy_model.return_value.loss, "val")

            module.log.assert_called_once_with(
                "val_perplexity", float("inf"), on_epoch=True, prog_bar=True, logger=True
            )
