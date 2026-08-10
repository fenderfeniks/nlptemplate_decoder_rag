import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.torch_utils import (
    BASE_SAFE_GLOBALS,
    _collect_lr_scheduler_globals,
    _collect_omegaconf_globals,
    _collect_transformers_globals,
    load_best_lora_weights,
    register_safe_globals,
)


class TestCollectors:
    def test_collect_omegaconf_globals_returns_list(self):
        """_collect_omegaconf_globals возвращает непустой список при наличии omegaconf."""
        result = _collect_omegaconf_globals()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_collect_lr_scheduler_globals_returns_list(self):
        """_collect_lr_scheduler_globals возвращает непустой список классов при наличии torch."""
        result = _collect_lr_scheduler_globals()
        assert isinstance(result, list)
        assert len(result) > 0
        # Каждый элемент — класс (не экземпляр)
        for item in result:
            assert isinstance(item, type)

    def test_collect_transformers_globals_returns_list(self):
        """_collect_transformers_globals возвращает непустой список при наличии transformers."""
        result = _collect_transformers_globals()
        assert isinstance(result, list)
        assert len(result) > 0

    # ------------------------------------------------------------------
    # Import-error fallback
    # ------------------------------------------------------------------

    def test_collectors_omegaconf_import_error(self):
        """_collect_omegaconf_globals возвращает [] при отсутствии omegaconf."""
        with patch.dict(sys.modules, {"omegaconf": None}):
            assert _collect_omegaconf_globals() == []

    def test_collectors_transformers_import_error(self):
        """_collect_transformers_globals возвращает [] при отсутствии transformers."""
        with patch.dict(sys.modules, {"transformers": None}):
            assert _collect_transformers_globals() == []

    def test_collectors_lr_scheduler_import_error(self):
        """_collect_lr_scheduler_globals возвращает [] при отсутствии torch.optim.lr_scheduler."""
        # Имитируем отсутствие модуля внутри torch
        with patch("src.utils.torch_utils._collect_lr_scheduler_globals") as mock_collect:
            mock_collect.return_value = []
            result = mock_collect()
        assert result == []


class TestRegisterSafeGlobals:
    @patch("src.utils.torch_utils.torch.serialization.add_safe_globals")
    def test_register_safe_globals_called_once(self, mock_add):
        """add_safe_globals вызывается ровно один раз."""
        register_safe_globals()
        mock_add.assert_called_once()

    @patch("src.utils.torch_utils.torch.serialization.add_safe_globals")
    def test_register_safe_globals_contains_base(self, mock_add):
        """Результирующий список содержит все объекты из BASE_SAFE_GLOBALS."""
        register_safe_globals()
        args = mock_add.call_args[0][0]
        for obj in BASE_SAFE_GLOBALS:
            assert obj in args

    @patch("src.utils.torch_utils.torch.serialization.add_safe_globals")
    def test_register_safe_globals_total_count(self, mock_add):
        """Итоговый список длиннее BASE_SAFE_GLOBALS (коллекторы дали хотя бы одно)."""
        register_safe_globals()
        args = mock_add.call_args[0][0]
        assert len(args) > len(BASE_SAFE_GLOBALS)

    @patch("src.utils.torch_utils.torch.serialization.add_safe_globals")
    def test_register_safe_globals_is_list(self, mock_add):
        """add_safe_globals получает список (не кортеж и не генератор)."""
        register_safe_globals()
        args = mock_add.call_args[0][0]
        assert isinstance(args, list)


class TestLoadBestLoraWeights:
    @patch("src.utils.torch_utils.torch.load")
    def test_load_with_state_dict_wrapper(self, mock_load):
        """Lightning-чекпоинт с ключом state_dict корректно распаковывается."""
        mock_load.return_value = {"state_dict": {"lora_A": 1.0}}
        mock_module = MagicMock()

        mock_peft = MagicMock()
        with patch.dict(sys.modules, {"peft": mock_peft}):
            load_best_lora_weights(mock_module, "dummy.ckpt")

        mock_load.assert_called_once_with("dummy.ckpt", map_location="cpu", weights_only=False)
        mock_peft.set_peft_model_state_dict.assert_called_once_with(
            mock_module.model, {"lora_A": 1.0}
        )

    @patch("src.utils.torch_utils.torch.load")
    def test_load_raw_state_dict(self, mock_load):
        """Сырой словарь весов (без обёртки state_dict) передаётся напрямую."""
        mock_load.return_value = {"lora_B": 2.0}
        mock_module = MagicMock()

        mock_peft = MagicMock()
        with patch.dict(sys.modules, {"peft": mock_peft}):
            load_best_lora_weights(mock_module, "raw.ckpt")

        mock_peft.set_peft_model_state_dict.assert_called_once_with(
            mock_module.model, {"lora_B": 2.0}
        )

    def test_import_error_without_peft(self):
        """ImportError при отсутствии peft содержит понятное сообщение."""
        with patch.dict(sys.modules, {"peft": None}):
            with pytest.raises(ImportError, match="необходима библиотека peft"):
                load_best_lora_weights(MagicMock(), "dummy.ckpt")

    @patch("src.utils.torch_utils.torch.load")
    def test_load_passes_cpu_map_location(self, mock_load):
        """Чекпоинт всегда загружается на CPU (map_location='cpu')."""
        mock_load.return_value = {}
        mock_module = MagicMock()

        mock_peft = MagicMock()
        with patch.dict(sys.modules, {"peft": mock_peft}):
            load_best_lora_weights(mock_module, "path.ckpt")

        _, kwargs = mock_load.call_args
        assert kwargs["map_location"] == "cpu"

    @patch("src.utils.torch_utils.torch.load")
    def test_set_peft_receives_module_model(self, mock_load):
        """set_peft_model_state_dict получает именно model_module.model."""
        mock_load.return_value = {"state_dict": {"k": "v"}}
        mock_module = MagicMock()
        sentinel_model = object()
        mock_module.model = sentinel_model

        mock_peft = MagicMock()
        with patch.dict(sys.modules, {"peft": mock_peft}):
            load_best_lora_weights(mock_module, "p.ckpt")

        first_arg = mock_peft.set_peft_model_state_dict.call_args[0][0]
        assert first_arg is sentinel_model
