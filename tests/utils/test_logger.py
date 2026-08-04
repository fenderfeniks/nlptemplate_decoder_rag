import logging
from unittest.mock import MagicMock, patch

from src.utils.logger import NOISY_LOGGERS, setup_logging


class TestLogger:
    @patch("src.utils.logger.os.getenv")
    def test_dev_environment(self, mock_getenv):
        """Проверка инициализации в dev-режиме (стандартный формат)."""
        mock_getenv.return_value = "dev"
        setup_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

        # Проверяем, что шумные логгеры подавлены
        for name in NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    @patch("src.utils.logger.os.getenv")
    def test_prod_environment(self, mock_getenv):
        """Проверка инициализации в prod-режиме."""
        mock_getenv.return_value = "prod"

        # Мокаем импорт jsonlogger, чтобы тест проходил без этой библиотеки
        with patch.dict("sys.modules", {"pythonjsonlogger": MagicMock()}):
            setup_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
