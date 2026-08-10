from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.utils.mlflow.adapter_saver import (
    _build_reg_model_name,
    _create_model_version,
    _patch_peft_config_for_hydra,
    _save_adapter_to_tempdir,
)


class TestAdapterSaver:
    def test_patch_peft_config_for_hydra(self):
        """Проверка, что ListConfig и DictConfig конвертируются в нативные типы."""
        mock_model = MagicMock()
        mock_peft_cfg = MagicMock()

        # Симулируем структуру, куда попали объекты Hydra
        mock_peft_cfg.target_modules = OmegaConf.create(["q_proj", "v_proj"])
        mock_peft_cfg.modules_to_save = OmegaConf.create({"emb": True})

        mock_model.peft_config = {"default": mock_peft_cfg}

        _patch_peft_config_for_hydra(mock_model)

        assert isinstance(mock_peft_cfg.target_modules, list)
        assert mock_peft_cfg.target_modules == ["q_proj", "v_proj"]

        assert isinstance(mock_peft_cfg.modules_to_save, dict)
        assert mock_peft_cfg.modules_to_save == {"emb": True}

    def test_build_reg_model_name(self):
        assert _build_reg_model_name("MyModel") == "MyModel_LoRA"
        assert _build_reg_model_name("MyModel", "QDoRA") == "MyModel_QDoRA"

    def test_save_adapter_to_tempdir_success(self, tmp_path):
        """Проверка успешного сохранения."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Мокаем save_pretrained, чтобы он создавал нужный файл
        def fake_save(path):
            (Path(path) / "adapter_config.json").touch()

        mock_model.save_pretrained.side_effect = fake_save

        _save_adapter_to_tempdir(mock_model, mock_tokenizer, tmp_path)

        mock_model.save_pretrained.assert_called_once_with(tmp_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(tmp_path)

    def test_save_adapter_to_tempdir_missing_config(self, tmp_path):
        """Если save_pretrained не создал файл, должна быть ошибка."""
        mock_model = MagicMock()  # Ничего не создает
        mock_tokenizer = MagicMock()

        with pytest.raises(FileNotFoundError, match="не создал adapter_config.json"):
            _save_adapter_to_tempdir(mock_model, mock_tokenizer, tmp_path)

    @patch("src.utils.mlflow.adapter_saver._register_model_version")
    def test_create_model_version_fallback(self, mock_register):
        """Проверка перехода к fallback-регистрации при ошибке create_model_version."""
        client = MagicMock()
        # Эмулируем ошибку при штатном создании версии
        client.create_model_version.side_effect = Exception("API error")
        mock_register.return_value = "2"

        version = _create_model_version(client, "run_123", "lora", "Model_LoRA")

        assert version == "2"
        mock_register.assert_called_once()
