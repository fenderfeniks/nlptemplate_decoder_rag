from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.utils.mlflow.adapter_loader import (
    _download_by_run_id,
    _find_adapter_config,
    resolve_lora_resume_path,
)


class TestAdapterLoader:
    def test_find_adapter_config_root(self, tmp_path):
        """Проверка поиска конфига прямо в корне."""
        (tmp_path / "adapter_config.json").touch()
        assert _find_adapter_config(tmp_path) == tmp_path

    def test_find_adapter_config_subdirs(self, tmp_path):
        """Проверка поиска конфига в стандартных подпапках."""
        peft_dir = tmp_path / "peft"
        peft_dir.mkdir()
        (peft_dir / "adapter_config.json").touch()
        assert _find_adapter_config(tmp_path) == peft_dir

    def test_find_adapter_config_rglob(self, tmp_path):
        """Проверка глубокого поиска, если конфиг запрятан глубоко."""
        deep_dir = tmp_path / "some" / "deep" / "path"
        deep_dir.mkdir(parents=True)
        (deep_dir / "adapter_config.json").touch()
        assert _find_adapter_config(tmp_path) == deep_dir

    def test_find_adapter_config_not_found(self, tmp_path):
        """Если конфига нет, возвращается None."""
        assert _find_adapter_config(tmp_path) is None

    @patch("src.utils.mlflow.adapter_loader.mlflow.artifacts.download_artifacts")
    def test_download_by_run_id(self, mock_download):
        mock_download.return_value = "/mock/path"
        result = _download_by_run_id("test_run", "lora_weights")

        mock_download.assert_called_once_with(run_id="test_run", artifact_path="lora_weights")
        assert result == Path("/mock/path")

    def test_resolve_lora_resume_path_disabled(self):
        """Если enabled=false или конфиг пуст, возвращается None."""
        assert resolve_lora_resume_path(None) is None
        assert resolve_lora_resume_path({"enabled": False}) is None

        cfg = OmegaConf.create({"enabled": False})
        assert resolve_lora_resume_path(cfg) is None

    @patch("src.utils.mlflow.adapter_loader._download_by_run_id")
    def test_resolve_lora_resume_path_missing_config_raises_error(self, mock_download, tmp_path):
        """Если после скачивания конфиг не найден, выбрасывается FileNotFoundError."""
        mock_download.return_value = tmp_path  # Пустая директория, без json

        cfg = {"enabled": True, "run_id": "123"}
        with pytest.raises(FileNotFoundError, match="adapter_config.json не найден"):
            resolve_lora_resume_path(cfg)
