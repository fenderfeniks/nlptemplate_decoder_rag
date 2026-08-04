from unittest.mock import MagicMock, patch

import pytest

from src.utils.checkpoint_utils import load_checkpoint


@pytest.fixture
def dummy_model():
    model = MagicMock()
    # Эмулируем возвращаемый объект load_state_dict
    result = MagicMock()
    result.missing_keys = []
    result.unexpected_keys = []
    model.load_state_dict.return_value = result
    return model


class TestCheckpointUtils:
    def test_missing_path(self, dummy_model):
        """Ошибка при отсутствии файла/директории."""
        with pytest.raises(FileNotFoundError, match="Указанный путь не существует"):
            load_checkpoint(dummy_model, "non_existent_path.bin")

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_load_single_file(self, mock_load, dummy_model, tmp_path):
        """Загрузка напрямую из файла .bin."""
        file_path = tmp_path / "model.bin"
        file_path.touch()

        mock_load.return_value = {"model.layer1": "weights"}

        load_checkpoint(dummy_model, file_path)

        mock_load.assert_called_once()
        dummy_model.load_state_dict.assert_called_once_with(
            {"model.layer1": "weights"}, strict=False
        )

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_load_pl_checkpoint(self, mock_load, dummy_model, tmp_path):
        """Разворачивание state_dict из PyTorch Lightning чекпоинта и обрезка префиксов."""
        file_path = tmp_path / "model.ckpt"
        file_path.touch()

        mock_load.return_value = {"state_dict": {"model.fc.weight": "weights"}}

        load_checkpoint(dummy_model, file_path)

        dummy_model.load_state_dict.assert_called_once_with({"fc.weight": "weights"}, strict=False)

    @patch("src.utils.checkpoint_utils.logger")
    @patch("src.utils.checkpoint_utils.torch.load")
    def test_logging_missing_keys(self, mock_load, mock_logger, dummy_model, tmp_path):
        """Проверка логирования missing/unexpected ключей."""
        file_path = tmp_path / "model.bin"
        file_path.touch()
        mock_load.return_value = {}

        # Эмулируем расхождения
        dummy_model.load_state_dict.return_value.missing_keys = ["layer2.weight"]
        dummy_model.load_state_dict.return_value.unexpected_keys = ["layer3.weight"]

        load_checkpoint(dummy_model, file_path)

        mock_logger.warning.assert_any_call(
            "Отсутствующие ключи в чекпоинте (%d шт.): %s", 1, ["layer2.weight"]
        )
        mock_logger.warning.assert_any_call(
            "Неожиданные ключи в чекпоинте (%d шт.): %s", 1, ["layer3.weight"]
        )
