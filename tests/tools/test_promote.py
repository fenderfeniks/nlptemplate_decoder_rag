from unittest.mock import MagicMock, patch

import pytest

from src.tools.promote import PromoteError, _promote


class TestPromoteModel:
    @patch("src.tools.promote.MlflowClient")
    @patch("src.tools.promote.mlflow")
    def test_promote_success_better_model(self, mock_mlflow, mock_client_cls):
        """Staging модель лучше Production (0.1 < 0.5) -> Успешный промоут."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Мокаем ответы MLflow: первый вызов для Staging, второй для Production
        mock_client.get_model_version_by_alias.side_effect = [
            MagicMock(version="2", tags={"val_loss": "0.1"}),
            MagicMock(version="1", tags={"val_loss": "0.5"}),
        ]

        _promote("dummy_uri", "my_model")

        # Проверяем, что алиас был обновлен
        mock_client.set_registered_model_alias.assert_called_once_with(
            "my_model", "Production", "2"
        )

    @patch("src.tools.promote.MlflowClient")
    @patch("src.tools.promote.mlflow")
    def test_promote_skipped_worse_model(self, mock_mlflow, mock_client_cls):
        """Staging модель хуже Production (0.6 > 0.5) -> Промоут отменен."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.get_model_version_by_alias.side_effect = [
            MagicMock(version="2", tags={"val_loss": "0.6"}),
            MagicMock(version="1", tags={"val_loss": "0.5"}),
        ]

        _promote("dummy_uri", "my_model")

        mock_client.set_registered_model_alias.assert_not_called()

    @patch("src.tools.promote.MlflowClient")
    @patch("src.tools.promote.mlflow")
    def test_promote_first_time(self, mock_mlflow, mock_client_cls):
        """Если Production модели еще нет, Staging должен стать Production."""
        from mlflow.exceptions import MlflowException

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def get_alias_mock(name, alias):
            if alias == "Staging":
                return MagicMock(version="1", tags={"val_loss": "0.3"})
            raise MlflowException("Alias not found")  # Имитация отсутствия Production

        mock_client.get_model_version_by_alias.side_effect = get_alias_mock

        _promote("dummy_uri", "my_model")

        mock_client.set_registered_model_alias.assert_called_once_with(
            "my_model", "Production", "1"
        )

    @patch("src.tools.promote.MlflowClient")
    @patch("src.tools.promote.mlflow")
    def test_promote_fails_no_staging(self, mock_mlflow, mock_client_cls):
        """Если нет Staging, скрипт должен упасть с понятной ошибкой."""
        from mlflow.exceptions import MlflowException

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_model_version_by_alias.side_effect = MlflowException("No Staging")

        with pytest.raises(PromoteError, match="Алиас 'Staging' не найден"):
            _promote("dummy_uri", "my_model")
