# tests/tools/test_promote.py
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException

from src.tools.promote import PromoteError, _promote


class TestPromoteModel:
    @pytest.fixture
    def mock_mlflow_client(self):
        with patch("src.tools.promote.MlflowClient") as mock_client_class:
            client = MagicMock()
            mock_client_class.return_value = client
            yield client

    def _make_mock_model_version(self, version: str, val_loss: float | None):
        mv = MagicMock()
        mv.version = version
        mv.tags = {"val_loss": str(val_loss)} if val_loss is not None else {}
        return mv

    def test_promotes_when_staging_is_better(self, mock_mlflow_client):
        """Если Staging (val_loss=1.5) лучше Production (val_loss=2.0), модель обновляется."""
        mock_mlflow_client.get_model_version_by_alias.side_effect = [
            self._make_mock_model_version("v2", 1.5),  # Staging
            self._make_mock_model_version("v1", 2.0),  # Production
        ]

        _promote("http://mlflow", "TestModel")

        # Проверяем, что алиас Production перевешен на версию v2
        mock_mlflow_client.set_registered_model_alias.assert_called_once_with(
            "TestModel", "Production", "v2"
        )

    def test_skips_when_staging_is_worse(self, mock_mlflow_client, caplog):
        """Если Staging хуже, алиас не меняется."""
        mock_mlflow_client.get_model_version_by_alias.side_effect = [
            self._make_mock_model_version("v2", 2.5),  # Staging
            self._make_mock_model_version("v1", 2.0),  # Production
        ]

        _promote("http://mlflow", "TestModel")

        mock_mlflow_client.set_registered_model_alias.assert_not_called()
        assert "хуже или равна текущей Production" in caplog.text

    def test_promotes_when_production_is_missing(self, mock_mlflow_client):
        """Если Production еще нет (ошибка MlflowException), Staging сразу становится Prod."""

        def mock_get_alias(name, alias):
            if alias == "Staging":
                return self._make_mock_model_version("v1", 1.5)
            raise MlflowException("Alias not found")

        mock_mlflow_client.get_model_version_by_alias.side_effect = mock_get_alias

        _promote("http://mlflow", "TestModel")

        mock_mlflow_client.set_registered_model_alias.assert_called_once_with(
            "TestModel", "Production", "v1"
        )

    def test_raises_error_if_staging_missing(self, mock_mlflow_client):
        """Если Staging нет, выбрасывается кастомная ошибка PromoteError."""
        mock_mlflow_client.get_model_version_by_alias.side_effect = MlflowException("Not found")

        with pytest.raises(PromoteError, match="Алиас 'Staging' не найден"):
            _promote("http://mlflow", "TestModel")

    def test_raises_error_if_no_val_loss_tag(self, mock_mlflow_client):
        """Если у Staging нет тега val_loss, оценить модель нельзя."""
        mock_mlflow_client.get_model_version_by_alias.return_value = self._make_mock_model_version(
            "v1", None
        )

        with pytest.raises(PromoteError, match="нет тега 'val_loss'"):
            _promote("http://mlflow", "TestModel")
