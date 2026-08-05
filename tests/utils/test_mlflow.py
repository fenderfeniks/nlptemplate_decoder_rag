# tests/utils/test_mlflow.py
import os
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.mlflow import (
    _build_reg_model_name,
    _ensure_tracking_uri,
    _find_adapter_config,
    _patch_peft_config_for_hydra,
    _strip_version_specifier,
    get_inference_pip_requirements,
    log_lora_to_mlflow,
    resolve_lora_resume_path,
)


class TestMLflowUtilsBasic:
    def test_strip_version_specifier(self):
        """Проверка удаления версий, маркеров окружения и extras."""
        assert _strip_version_specifier("torch>=2.0") == "torch"
        assert _strip_version_specifier("numpy==1.21.0") == "numpy"
        assert _strip_version_specifier("vllm[tensorizer]==0.4.0") == "vllm"
        assert _strip_version_specifier("peft; sys_platform == 'linux'") == "peft"

    def test_build_reg_model_name(self):
        """Проверка сборки имени модели для Registry."""
        assert _build_reg_model_name("Llama-3", "LoRA") == "Llama-3_LoRA"
        assert _build_reg_model_name("Mistral", "Full") == "Mistral_Full"

    @patch("src.utils.mlflow.version")
    @patch("src.utils.mlflow.tomllib.load")
    @patch("builtins.open")
    def test_get_inference_pip_requirements(self, mock_open, mock_toml, mock_version):
        """Парсинг зависимостей инференса из pyproject.toml."""
        mock_toml.return_value = {
            "project": {
                "optional-dependencies": {"inference-core": ["vllm[tensorizer]>=0.4.0", "fastapi"]}
            }
        }

        def side_effect(pkg_name):
            if pkg_name == "vllm":
                return "0.4.1"
            if pkg_name == "fastapi":
                return "0.100.0"
            raise Exception("Unknown")

        mock_version.side_effect = side_effect
        reqs = get_inference_pip_requirements("pyproject.toml")

        assert len(reqs) == 2
        assert "vllm==0.4.1" in reqs
        assert "fastapi==0.100.0" in reqs


class TestMLflowResumeAndConfig:
    def test_ensure_tracking_uri_env(self):
        """Проверка чтения URI из переменных окружения."""
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://test-uri"}):
            with patch("src.utils.mlflow.mlflow.set_tracking_uri") as mock_set_uri:
                _ensure_tracking_uri()
                mock_set_uri.assert_called_once_with("http://test-uri")

    def test_find_adapter_config(self, tmp_path):
        """Проверка поиска adapter_config.json во вложенных папках."""
        # Создаем вложенную структуру
        target_dir = tmp_path / "peft" / "model"
        target_dir.mkdir(parents=True)
        (target_dir / "adapter_config.json").touch()

        # Должен найти папку, содержащую файл
        res = _find_adapter_config(tmp_path)
        assert res == target_dir

    def test_patch_peft_config_for_hydra(self):
        """Проверка конвертации ListConfig/DictConfig в нативные типы для PEFT."""

        class DummyPeftConfig:
            def __init__(self):
                self.target_modules = ListConfig(["q_proj", "v_proj"])
                self.kwargs = DictConfig({"alpha": 16})
                self.string_val = "test"

        model = MagicMock()
        model.peft_config = {"default": DummyPeftConfig()}

        _patch_peft_config_for_hydra(model)

        cfg = model.peft_config["default"]
        assert isinstance(cfg.target_modules, list)
        assert isinstance(cfg.kwargs, dict)
        assert cfg.string_val == "test"

    def test_resolve_lora_resume_path_disabled(self):
        """Если resume выключен, функция сразу возвращает None."""
        assert resolve_lora_resume_path({"enabled": False}) is None
        assert resolve_lora_resume_path(None) is None

    @patch("src.utils.mlflow.mlflow")
    def test_resolve_lora_resume_path_run_id(self, mock_mlflow, tmp_path):
        """Успешное скачивание по run_id."""
        adapter_dir = tmp_path / "peft"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").touch()

        mock_mlflow.artifacts.download_artifacts.return_value = str(adapter_dir)

        cfg = {"enabled": True, "run_id": "run123", "artifact_path": "lora"}
        res = resolve_lora_resume_path(cfg)

        assert res == str(adapter_dir)
        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run123", artifact_path="lora"
        )


class TestMLflowLoggingLogic:
    @patch("src.utils.mlflow._save_adapter_to_tempdir")
    @patch("src.utils.mlflow._register_model_version")
    @patch("src.utils.mlflow.MlflowClient")
    @patch("src.utils.mlflow.mlflow")
    def test_log_lora_to_mlflow_full_pipeline(
        self, mock_mlflow, mock_client_cls, mock_reg_model, mock_save
    ):
        """Проверка полного пайплайна сохранения, регистрации и алиасов."""
        mock_reg_model.return_value = "v1"
        mock_client = mock_client_cls.return_value

        mock_client.create_model_version.return_value.version = "v1"

        cfg = OmegaConf.create(
            {
                "decoder_pipeline": {"model": {"architecture": {"mlflow_model_name": "TestModel"}}},
                "logger": {
                    "registry": {
                        "artifact_path": "lora_weights",
                        "register_on_success": True,
                        "promote_to_staging": True,
                    }
                },
            }
        )

        log_lora_to_mlflow(
            cfg,
            model_module=MagicMock(),
            tokenizer=MagicMock(),
            run_id="run123",
            pipeline_name="decoder_pipeline",
            best_score=0.15,
        )

        mock_save.assert_called_once()
        mock_mlflow.log_artifacts.assert_called_once()
        mock_mlflow.log_metric.assert_called_once_with("promotion_candidate_val_loss", 0.15)

        # Проверяем, что алиасы и теги навесились
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="TestModel_LoRA", alias="Staging", version="v1"
        )
        mock_client.set_model_version_tag.assert_called_once_with(
            "TestModel_LoRA", "v1", "val_loss", "0.15"
        )

    @patch("src.utils.mlflow._save_adapter_to_tempdir")
    @patch("src.utils.mlflow._register_model_version")
    @patch("src.utils.mlflow.mlflow")
    def test_log_lora_to_mlflow_skip_registration(self, mock_mlflow, mock_reg_model, mock_save):
        """Если register_on_success=False, пайплайн прерывается после загрузки артефактов."""
        cfg = OmegaConf.create(
            {
                "decoder_pipeline": {"model": {"architecture": {"mlflow_model_name": "TestModel"}}},
                "logger": {"registry": {"register_on_success": False}},
            }
        )

        log_lora_to_mlflow(
            cfg,
            model_module=MagicMock(),
            tokenizer=MagicMock(),
            run_id="run123",
            pipeline_name="decoder_pipeline",
        )

        mock_save.assert_called_once()
        mock_mlflow.log_artifacts.assert_called_once()
        mock_reg_model.assert_not_called()
