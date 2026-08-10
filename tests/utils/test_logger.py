import logging
from unittest.mock import MagicMock, patch

import pytest

from src.utils.logger import NOISY_LOGGERS, setup_logging


class TestLogger:
    def setup_method(self):
        """Сбрасываем root logger перед каждым тестом."""
        root = logging.getLogger()
        root.handlers.clear()

    # ------------------------------------------------------------------
    # dev-режим
    # ------------------------------------------------------------------

    @patch("src.utils.logger.os.getenv")
    def test_dev_environment(self, mock_getenv):
        """В dev-режиме создаётся ровно один StreamHandler."""
        mock_getenv.return_value = "dev"
        setup_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    @patch("src.utils.logger.os.getenv")
    def test_dev_formatter_contains_lineno(self, mock_getenv):
        """В dev-режиме форматтер содержит %(lineno)d."""
        mock_getenv.return_value = "dev"
        setup_logging()

        handler = logging.getLogger().handlers[0]
        assert "lineno" in handler.formatter._fmt

    @patch("src.utils.logger.os.getenv")
    def test_noisy_loggers_suppressed(self, mock_getenv):
        """Шумные логгеры подавляются до WARNING независимо от окружения."""
        mock_getenv.return_value = "dev"
        setup_logging()

        for name in NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    @patch("src.utils.logger.os.getenv")
    def test_default_level_applied(self, mock_getenv):
        """Уровень root logger устанавливается из аргумента default_level."""
        mock_getenv.return_value = "dev"
        setup_logging(default_level=logging.DEBUG)

        assert logging.getLogger().level == logging.DEBUG

    @patch("src.utils.logger.os.getenv")
    def test_repeated_calls_dont_stack_handlers(self, mock_getenv):
        """Повторный вызов setup_logging не накапливает хэндлеры."""
        mock_getenv.return_value = "dev"
        setup_logging()
        setup_logging()

        assert len(logging.getLogger().handlers) == 1

    # ------------------------------------------------------------------
    # prod-режим
    # ------------------------------------------------------------------

    @patch("src.utils.logger.os.getenv")
    def test_prod_environment_with_jsonlogger(self, mock_getenv):
        """В prod-режиме используется JsonFormatter, если библиотека доступна."""
        mock_getenv.return_value = "prod"

        mock_json_formatter = MagicMock()
        mock_jsonlogger_module = MagicMock()
        mock_jsonlogger_module.JsonFormatter.return_value = mock_json_formatter

        mock_pythonjsonlogger = MagicMock()
        mock_pythonjsonlogger.jsonlogger = mock_jsonlogger_module

        with patch.dict("sys.modules", {"pythonjsonlogger": mock_pythonjsonlogger}):
            setup_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        # JsonFormatter был создан
        mock_jsonlogger_module.JsonFormatter.assert_called_once()

    @patch("src.utils.logger.os.getenv")
    def test_prod_environment_fallback_without_jsonlogger(self, mock_getenv):
        """В prod-режиме без pythonjsonlogger — fallback на стандартный Formatter."""
        mock_getenv.return_value = "prod"

        with patch.dict("sys.modules", {"pythonjsonlogger": None}):
            setup_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        # Упал на обычный Formatter — он не JsonFormatter (нет атрибута .supported_keys)
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, logging.Formatter)

    @patch("src.utils.logger.os.getenv")
    def test_prod_noisy_loggers_suppressed(self, mock_getenv):
        """В prod-режиме шумные логгеры тоже подавлены."""
        mock_getenv.return_value = "prod"

        with patch.dict("sys.modules", {"pythonjsonlogger": None}):
            setup_logging()

        for name in NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING
