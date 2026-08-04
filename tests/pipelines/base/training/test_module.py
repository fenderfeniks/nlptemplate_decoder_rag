import sys
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from src.pipelines.base.training.module import OptimizerMixin


class DummyLightningModule(OptimizerMixin):
    """Фейковый модуль для тестирования миксина без инициализации всего Lightning."""

    def __init__(self, model, optimizer_cfg, scheduler_cfg=None):
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        # Мокаем trainer, который обычно предоставляет pl.LightningModule
        self.trainer = MagicMock()


class TestOptimizerMixinCheckpointing:
    def test_on_save_checkpoint_peft_model(self):
        """Проверка, что для PeftModel сохраняются только веса адаптера."""
        # 1. Мокаем библиотеку peft
        mock_peft = MagicMock()
        mock_peft_model_class = type("PeftModel", (object,), {})
        mock_peft.PeftModel = mock_peft_model_class
        mock_peft.utils = MagicMock()
        mock_peft.utils.get_peft_model_state_dict.return_value = {"peft_weight": 1}

        # 2. Подменяем импорты на лету
        with patch.dict(sys.modules, {"peft": mock_peft, "peft.utils": mock_peft.utils}):
            # Создаем фейковую модель, которая пройдет проверку isinstance
            class FakePeftModel(mock_peft_model_class):
                pass

            module = DummyLightningModule(model=FakePeftModel(), optimizer_cfg=None)

            # Исходный чекпоинт с полной массой весов
            checkpoint = {"state_dict": {"full_weight": 2}}

            module.on_save_checkpoint(checkpoint)

            mock_peft.utils.get_peft_model_state_dict.assert_called_once()
            # Проверяем, что исходный словарь заменился на словарь PEFT
            assert checkpoint["state_dict"] == {"peft_weight": 1}

    def test_on_save_checkpoint_normal_model(self):
        """Для обычной модели (Full FT) чекпоинт должен остаться без изменений."""
        module = DummyLightningModule(model=nn.Linear(1, 1), optimizer_cfg=None)
        checkpoint = {"state_dict": {"weight": 1}}

        module.on_save_checkpoint(checkpoint)

        assert checkpoint["state_dict"] == {"weight": 1}

    def test_on_save_checkpoint_no_peft_installed(self):
        """Если peft не установлен, метод должен безопасно проигнорировать работу."""
        module = DummyLightningModule(model=nn.Linear(1, 1), optimizer_cfg=None)
        checkpoint = {"state_dict": {"weight": 1}}

        with patch.dict(sys.modules, {"peft": None}):
            module.on_save_checkpoint(checkpoint)

        assert checkpoint["state_dict"] == {"weight": 1}


class TestOptimizerMixinConfigureOptimizers:
    def test_no_trainable_params_warning(self, caplog):
        """Проверка ворнинга, если все параметры модели заморожены."""
        model = nn.Linear(1, 1)
        for p in model.parameters():
            p.requires_grad = False

        module = DummyLightningModule(model, optimizer_cfg=MagicMock(return_value="opt"))

        with caplog.at_level("WARNING"):
            module.configure_optimizers()

        assert "Нет параметров для обучения!" in caplog.text

    def test_optimizer_callable_no_scheduler(self):
        """Проверка работы с callable оптимизатором и без шедулера."""
        model = nn.Linear(1, 1)
        mock_opt_callable = MagicMock(return_value="my_optimizer")
        module = DummyLightningModule(model, optimizer_cfg=mock_opt_callable, scheduler_cfg=None)

        res = module.configure_optimizers()

        mock_opt_callable.assert_called_once()
        # Если шедулера нет, возвращается просто оптимизатор
        assert res == "my_optimizer"

    @patch("src.pipelines.base.training.module.instantiate")
    def test_optimizer_and_scheduler_from_config(self, mock_instantiate):
        """Проверка инстанцирования оптимизатора и шедулера через Hydra."""
        model = nn.Linear(1, 1)
        module = DummyLightningModule(
            model=model, optimizer_cfg={"_target_": "Opt"}, scheduler_cfg={"_target_": "Sched"}
        )

        mock_instantiate.side_effect = ["my_optimizer", "my_scheduler"]

        res = module.configure_optimizers()

        assert res == {
            "optimizer": "my_optimizer",
            "lr_scheduler": {
                "scheduler": "my_scheduler",
                "interval": "step",
                "frequency": 1,
            },
        }

    def test_scheduler_callable_success(self):
        """Проверка callable шедулера: должен передаваться total_steps от trainer."""
        model = nn.Linear(1, 1)
        mock_opt = MagicMock(return_value="my_optimizer")
        mock_sched = MagicMock(return_value="my_scheduler")

        module = DummyLightningModule(model, optimizer_cfg=mock_opt, scheduler_cfg=mock_sched)
        module.trainer.estimated_stepping_batches = 100

        res = module.configure_optimizers()

        mock_sched.assert_called_once_with(optimizer="my_optimizer", num_training_steps=100)
        assert res["lr_scheduler"]["scheduler"] == "my_scheduler"

    def test_scheduler_callable_inf_steps_raises_error(self):
        """Если trainer.estimated_stepping_batches равно inf, должна быть ошибка."""
        model = nn.Linear(1, 1)
        module = DummyLightningModule(model, optimizer_cfg=MagicMock(), scheduler_cfg=MagicMock())
        module.trainer.estimated_stepping_batches = float("inf")

        with pytest.raises(ValueError, match="задайте max_steps в конфиге тренера"):
            module.configure_optimizers()
