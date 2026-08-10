from unittest.mock import MagicMock, patch

from src.utils.mlflow.runner import extract_mlflow_run_id


class TestRunner:
    def test_extract_mlflow_run_id_from_logger(self):
        """Проверка извлечения run_id из разных атрибутов логгера."""
        trainer = MagicMock()

        # Вариант 1: атрибут run_id
        trainer.logger.run_id = "run_1"
        assert extract_mlflow_run_id(trainer) == "run_1"

        # Вариант 2: атрибут _run_id (очищаем предыдущий)
        del trainer.logger.run_id
        trainer.logger._run_id = "run_2"
        assert extract_mlflow_run_id(trainer) == "run_2"

    def test_extract_mlflow_run_id_no_logger(self):
        """Если логгера нет, возвращается None (без падений)."""
        trainer = MagicMock()
        trainer.logger = None
        assert extract_mlflow_run_id(trainer) is None

    @patch("src.utils.mlflow.runner.mlflow.active_run")
    def test_extract_mlflow_run_id_fallback_to_active_run(self, mock_active_run):
        """Если логгер пустой, должен быть fallback на mlflow.active_run()."""
        trainer = MagicMock()
        # Логгер есть, но атрибутов run_id в нем нет
        del trainer.logger.run_id
        del trainer.logger._run_id
        del trainer.logger.runid

        mock_active = MagicMock()
        mock_active.info.run_id = "active_123"
        mock_active_run.return_value = mock_active

        assert extract_mlflow_run_id(trainer) == "active_123"
        mock_active_run.assert_called_once()
