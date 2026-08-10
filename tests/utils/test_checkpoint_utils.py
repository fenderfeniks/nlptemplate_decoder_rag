from unittest.mock import MagicMock, patch

import pytest

from src.utils.checkpoint_utils import load_checkpoint


@pytest.fixture
def dummy_model():
    model = MagicMock()
    result = MagicMock()
    result.missing_keys = []
    result.unexpected_keys = []
    model.load_state_dict.return_value = result
    return model


class TestCheckpointUtils:
    # ------------------------------------------------------------------
    # Базовые сценарии (уже были)
    # ------------------------------------------------------------------

    def test_missing_path(self, dummy_model):
        """Ошибка при отсутствии файла/директории."""
        with pytest.raises(FileNotFoundError, match="Указанный путь не существует"):
            load_checkpoint(dummy_model, "non_existent_path.bin")

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_load_single_file(self, mock_load, dummy_model, tmp_path):
        """Загрузка напрямую из файла .bin (без обёртки state_dict)."""
        file_path = tmp_path / "model.bin"
        file_path.touch()

        # Ключ БЕЗ state_dict — словарь передаётся as-is
        mock_load.return_value = {"layer1.weight": "weights"}

        load_checkpoint(dummy_model, file_path)

        mock_load.assert_called_once()
        dummy_model.load_state_dict.assert_called_once_with(
            {"layer1.weight": "weights"}, strict=False
        )

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_load_pl_checkpoint(self, mock_load, dummy_model, tmp_path):
        """Разворачивание state_dict из PL-чекпоинта и обрезка префикса 'model.'."""
        file_path = tmp_path / "model.ckpt"
        file_path.touch()

        mock_load.return_value = {"state_dict": {"model.fc.weight": "weights"}}

        load_checkpoint(dummy_model, file_path)

        dummy_model.load_state_dict.assert_called_once_with({"fc.weight": "weights"}, strict=False)

    @patch("src.utils.checkpoint_utils.logger")
    @patch("src.utils.checkpoint_utils.torch.load")
    def test_logging_missing_and_unexpected_keys(
        self, mock_load, mock_logger, dummy_model, tmp_path
    ):
        """Проверка логирования missing/unexpected ключей."""
        file_path = tmp_path / "model.bin"
        file_path.touch()
        mock_load.return_value = {}

        dummy_model.load_state_dict.return_value.missing_keys = ["layer2.weight"]
        dummy_model.load_state_dict.return_value.unexpected_keys = ["layer3.weight"]

        load_checkpoint(dummy_model, file_path)

        mock_logger.warning.assert_any_call(
            "Отсутствующие ключи в чекпоинте (%d шт.): %s", 1, ["layer2.weight"]
        )
        mock_logger.warning.assert_any_call(
            "Неожиданные ключи в чекпоинте (%d шт.): %s", 1, ["layer3.weight"]
        )

    # ------------------------------------------------------------------
    # Загрузка из директории с pytorch_model.bin
    # ------------------------------------------------------------------

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_load_from_directory_with_bin(self, mock_load, dummy_model, tmp_path):
        """Директория содержит pytorch_model.bin — загружаем из него."""
        (tmp_path / "pytorch_model.bin").touch()
        mock_load.return_value = {"fc.weight": "w"}

        result = load_checkpoint(dummy_model, tmp_path)

        mock_load.assert_called_once()
        # Первый позиционный аргумент — путь к файлу
        assert mock_load.call_args[0][0] == tmp_path / "pytorch_model.bin"
        dummy_model.load_state_dict.assert_called_once_with({"fc.weight": "w"}, strict=False)
        assert result is dummy_model

    # ------------------------------------------------------------------
    # Директория без нужных файлов
    # ------------------------------------------------------------------

    def test_directory_without_known_files_raises(self, dummy_model, tmp_path):
        """Директория есть, но нет ни adapter_config.json ни pytorch_model.bin."""
        with pytest.raises(FileNotFoundError, match="не найден ни adapter_config.json"):
            load_checkpoint(dummy_model, tmp_path)

    # ------------------------------------------------------------------
    # LoRA / PEFT
    # ------------------------------------------------------------------

    def test_load_lora_adapter(self, dummy_model, tmp_path):
        """Директория с adapter_config.json → загружается как PeftModel."""
        (tmp_path / "adapter_config.json").touch()

        mock_peft_model = MagicMock()
        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel.from_pretrained.return_value = mock_peft_model

        with patch.dict("sys.modules", {"peft": mock_peft_module}):
            result = load_checkpoint(dummy_model, tmp_path)

        mock_peft_module.PeftModel.from_pretrained.assert_called_once_with(
            dummy_model, str(tmp_path)
        )
        assert result is mock_peft_model

    def test_load_lora_adapter_import_error(self, dummy_model, tmp_path):
        """Если peft не установлен — выбрасывается ImportError."""
        (tmp_path / "adapter_config.json").touch()

        with patch.dict("sys.modules", {"peft": None}):
            with pytest.raises(ImportError, match="необходима библиотека peft"):
                load_checkpoint(dummy_model, tmp_path)

    # ------------------------------------------------------------------
    # PL-чекпоинт: ключи без префикса model. не теряются
    # ------------------------------------------------------------------

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_pl_checkpoint_key_without_model_prefix(self, mock_load, dummy_model, tmp_path):
        """removeprefix не трогает ключи, которые не начинаются с 'model.'."""
        file_path = tmp_path / "model.ckpt"
        file_path.touch()

        mock_load.return_value = {
            "state_dict": {
                "model.encoder.weight": "w1",
                "other_key": "w2",  # нет префикса model.
            }
        }

        load_checkpoint(dummy_model, file_path)

        dummy_model.load_state_dict.assert_called_once_with(
            {"encoder.weight": "w1", "other_key": "w2"},
            strict=False,
        )

    # ------------------------------------------------------------------
    # Возвращаемое значение и логирование при чистой загрузке
    # ------------------------------------------------------------------

    @patch("src.utils.checkpoint_utils.logger")
    @patch("src.utils.checkpoint_utils.torch.load")
    def test_clean_load_logs_no_discrepancies(self, mock_load, mock_logger, dummy_model, tmp_path):
        """При отсутствии расхождений пишем info, а не warning."""
        file_path = tmp_path / "model.bin"
        file_path.touch()
        mock_load.return_value = {}

        load_checkpoint(dummy_model, file_path)

        # warning не должен вызываться вообще
        mock_logger.warning.assert_not_called()
        # info вызывался хотя бы раз (последнее сообщение "без расхождений")
        info_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("без расхождений" in m for m in info_messages)

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_returns_model(self, mock_load, dummy_model, tmp_path):
        """load_checkpoint возвращает модель (ту же самую при не-LoRA загрузке)."""
        file_path = tmp_path / "weights.pt"
        file_path.touch()
        mock_load.return_value = {}

        result = load_checkpoint(dummy_model, file_path)

        assert result is dummy_model

    # ------------------------------------------------------------------
    # device пробрасывается в torch.load
    # ------------------------------------------------------------------

    @patch("src.utils.checkpoint_utils.torch.load")
    def test_device_passed_to_torch_load(self, mock_load, dummy_model, tmp_path):
        """Аргумент device корректно прокидывается в torch.load."""
        file_path = tmp_path / "model.bin"
        file_path.touch()
        mock_load.return_value = {}

        load_checkpoint(dummy_model, file_path, device="cuda")

        _, kwargs = mock_load.call_args
        assert kwargs.get("map_location") == "cuda"
