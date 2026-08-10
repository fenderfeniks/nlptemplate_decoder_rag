"""Тесты для src/utils/hydra_utils.py.

Тяжёлые зависимости (OmegaConf, Hydra, схемы) мокируются там, где они
требуют полноценной инициализации конфига.  Функции, работающие только
с DictConfig/ListConfig, тестируются напрямую через реальные объекты OmegaConf.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.hydra_utils import (
    _force_utf8_console_encoding,
    _resolve_training_callbacks,
)


# ---------------------------------------------------------------------------
# _force_utf8_console_encoding
# ---------------------------------------------------------------------------


class TestForceUtf8ConsoleEncoding:
    def test_reconfigures_stream_handler(self):
        """reconfigure вызывается на StreamHandler с поддержкой этого метода."""
        mock_stream = MagicMock()
        mock_stream.reconfigure = MagicMock()

        handler = logging.StreamHandler()
        handler.stream = mock_stream

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _force_utf8_console_encoding()
            mock_stream.reconfigure.assert_called_once_with(
                encoding="utf-8", errors="backslashreplace"
            )
        finally:
            root.removeHandler(handler)

    def test_skips_handler_without_reconfigure(self):
        """Хэндлер без метода reconfigure не вызывает ошибки."""
        mock_stream = MagicMock(spec=[])  # нет reconfigure
        handler = logging.StreamHandler()
        handler.stream = mock_stream

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _force_utf8_console_encoding()  # не должен упасть
        finally:
            root.removeHandler(handler)

    def test_handles_value_error_gracefully(self):
        """ValueError внутри reconfigure перехватывается без пробрасывания."""
        mock_stream = MagicMock()
        mock_stream.reconfigure.side_effect = ValueError("bad encoding")

        handler = logging.StreamHandler()
        handler.stream = mock_stream

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _force_utf8_console_encoding()  # не должен падать
        finally:
            root.removeHandler(handler)

    def test_handles_os_error_gracefully(self):
        """OSError внутри reconfigure перехватывается без пробрасывания."""
        mock_stream = MagicMock()
        mock_stream.reconfigure.side_effect = OSError("stream closed")

        handler = logging.StreamHandler()
        handler.stream = mock_stream

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _force_utf8_console_encoding()
        finally:
            root.removeHandler(handler)


# ---------------------------------------------------------------------------
# _resolve_training_callbacks
# ---------------------------------------------------------------------------


class TestResolveTrainingCallbacks:
    def _make_training_cfg(self, callbacks_value) -> DictConfig:
        """Создаёт DictConfig с нужным значением callbacks."""
        cfg = OmegaConf.create({"callbacks": callbacks_value})
        OmegaConf.set_struct(cfg, False)
        return cfg

    # -- DictConfig → список -----------------------------------------------

    def test_dict_config_converted_to_list(self):
        """callbacks как DictConfig → конвертируется в список."""
        cfg = self._make_training_cfg({"ckpt": {"monitor": "val_loss"}, "lr": {"log_every": 1}})
        result = _resolve_training_callbacks(cfg)
        assert isinstance(result.callbacks, ListConfig)
        assert len(result.callbacks) == 2

    # -- ListConfig сохраняется --------------------------------------------

    def test_list_config_preserved(self):
        """callbacks уже ListConfig → остаётся ListConfig той же длины."""
        cfg = OmegaConf.create({"callbacks": [{"monitor": "val_loss"}, {"log": True}]})
        OmegaConf.set_struct(cfg, False)
        result = _resolve_training_callbacks(cfg)
        assert len(result.callbacks) == 2

    # -- None / пустой → пустой список -------------------------------------

    def test_none_callbacks_becomes_empty_list(self):
        """callbacks=None → пустой список."""
        cfg = OmegaConf.create({"callbacks": None})
        OmegaConf.set_struct(cfg, False)
        result = _resolve_training_callbacks(cfg)
        assert result.callbacks == [] or list(result.callbacks) == []

    def test_missing_callbacks_becomes_empty_list(self):
        """Отсутствующий ключ callbacks → пустой список."""
        cfg = OmegaConf.create({})
        OmegaConf.set_struct(cfg, False)
        result = _resolve_training_callbacks(cfg)
        assert list(result.callbacks) == []

    # -- Неверный тип → TypeError ------------------------------------------

    def test_wrong_type_raises_type_error(self):
        """callbacks — строка → TypeError с подсказкой."""
        # Нужно создать через struct=False, чтобы протолкнуть строку как целый узел
        cfg = OmegaConf.create({"training": {"callbacks": "bad_value"}})
        training_node = cfg.training
        OmegaConf.set_struct(training_node, False)

        # Заменяем callbacks на python-строку напрямую через dict
        raw = OmegaConf.to_container(training_node, resolve=True)
        raw["callbacks"] = "invalid_string"
        training_cfg = OmegaConf.create(raw)
        OmegaConf.set_struct(training_cfg, False)

        with pytest.raises(TypeError, match="DictConfig или ListConfig"):
            _resolve_training_callbacks(training_cfg)

    # -- Строки внутри списка → ValueError ---------------------------------

    def test_string_items_raise_value_error(self):
        """Строки вместо конфигов коллбэков → ValueError с подсказкой."""
        cfg = OmegaConf.create({"callbacks": ["checkpoint", "lr_monitor"]})
        OmegaConf.set_struct(cfg, False)

        with pytest.raises(ValueError, match="строки вместо конфигов"):
            _resolve_training_callbacks(cfg)


# ---------------------------------------------------------------------------
# setup_config (интеграционный; тяжёлые зависимости мокируются)
# ---------------------------------------------------------------------------


class TestSetupConfig:
    def _make_minimal_cfg(self) -> DictConfig:
        """Минимальный конфиг, проходящий через setup_config."""
        return OmegaConf.create(
            {
                "paths": {"root_dir": "", "log_dir": "/tmp/logs"},
            }
        )

    @patch("src.utils.hydra_utils._force_utf8_console_encoding")
    @patch("src.utils.hydra_utils.OmegaConf.structured")
    @patch("src.utils.hydra_utils.OmegaConf.merge")
    def test_setup_config_calls_utf8_encoding(self, mock_merge, mock_structured, mock_utf8):
        """setup_config вызывает _force_utf8_console_encoding."""
        from src.utils.hydra_utils import setup_config

        cfg = self._make_minimal_cfg()
        mock_structured.return_value = OmegaConf.create({})
        mock_merge.return_value = cfg

        setup_config(cfg)

        mock_utf8.assert_called_once()

    @patch("src.utils.hydra_utils._force_utf8_console_encoding")
    @patch("src.utils.hydra_utils.OmegaConf.structured")
    @patch("src.utils.hydra_utils.OmegaConf.merge")
    def test_setup_config_removes_meta_keys(self, mock_merge, mock_structured, mock_utf8):
        """setup_config удаляет _self_ и defaults из конфига."""
        from src.utils.hydra_utils import setup_config

        cfg = self._make_minimal_cfg()
        cfg["_self_"] = "x"
        cfg["defaults"] = ["base"]
        mock_structured.return_value = OmegaConf.create({})
        mock_merge.return_value = cfg

        setup_config(cfg)

        assert "_self_" not in cfg
        assert "defaults" not in cfg
