# tests/tools/test_merge_lora.py
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from src.tools.merge_lora import merge_and_export


class TestMergeLoraTool:
    @patch("src.tools.merge_lora.PeftModel")
    @patch("src.tools.merge_lora.resolve_lora_resume_path")
    @patch("src.tools.merge_lora.hydra.utils.instantiate")
    @patch("src.tools.merge_lora.setup_config")
    def test_merge_and_export_success(
        self, mock_setup_config, mock_instantiate, mock_resolve_path, mock_peft, tmp_path
    ):
        out_dir = tmp_path / "models"
        cfg = OmegaConf.create(
            {
                "pipeline_name": "test_pipeline",
                "test_pipeline": {
                    "model": {
                        "tokenizer": {},
                        "builder": {"model_name_or_path": "my-model"},
                        "architecture": {"mlflow_model_name": "TestModel"},
                    }
                },
                "logger": {
                    "pylightning": {"tracking_uri": "http"},
                    "registry": {"artifact_path": "model"},
                },
                "paths": {"model_dir": str(out_dir)},
            }
        )
        mock_setup_config.return_value = cfg
        mock_resolve_path.return_value = "fake/path"

        mock_tokenizer = MagicMock()
        mock_builder = MagicMock()
        mock_instantiate.side_effect = [mock_tokenizer, mock_builder]

        # Выстраиваем корректную цепочку моков
        mock_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged_model
        mock_peft.from_pretrained.return_value = mock_model

        merge_and_export.__wrapped__(cfg)

        expected_path = out_dir / "merged_my-model"
        mock_merged_model.save_pretrained.assert_called_once_with(expected_path)
