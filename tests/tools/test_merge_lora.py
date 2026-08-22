import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

# Укажи правильный путь импорта. Предполагаю, что файл называется merge_lora.py
from src.tools.merge_lora import merge_and_export


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def base_cfg():
    return OmegaConf.create(
        {
            "pipeline_name": "sequence_pipeline",
            "model": {
                "architecture": {"mlflow_model_name": "MyModel"},
                "tokenizer": {"_target_": "dummy_tokenizer"},
                "builder": {"_target_": "dummy_builder", "modifiers": ["some_modifier"]},
            },
            "system": {
                "logger": {"experiment_logger": {"_target_": "dummy"}},
                "storage": {"_target_": "dummy", "uri_prefix": "s3://bucket/"},
                "storage_router": {"_target_": "dummy"},
                "manifest": {"uri": "s3://bucket/manifest.json"},
                "paths": {"model_dir": "/tmp/models"},
            },
        }
    )


@pytest.fixture
def mock_instantiate(mocker):
    return mocker.patch("src.tools.merge_lora.hydra.utils.instantiate")


@pytest.fixture
def mock_setup_config(mocker):
    mock = mocker.patch("src.tools.merge_lora.setup_config")
    mock.side_effect = lambda x: x
    return mock


@pytest.fixture
def mock_sys_exit(mocker):
    return mocker.patch("src.tools.merge_lora.sys.exit")


@pytest.fixture
def mock_resolver(mocker):
    return mocker.patch("src.tools.merge_lora.ArtifactResolver")


@pytest.fixture
def mock_peft(mocker):
    return mocker.patch("src.tools.merge_lora.PeftModel")


@pytest.fixture
def mock_torch(mocker):
    """Мокает PyTorch, чтобы вызов empty_cache не падал на машинах без GPU."""
    return mocker.patch("src.tools.merge_lora.torch")


# ===========================================================================
# Тесты
# ===========================================================================


class TestMergeAndExport:
    def test_missing_mlflow_name(self, base_cfg, mock_setup_config):
        """Ошибка при отсутствии mlflow_model_name в конфиге."""
        base_cfg.model.architecture.pop("mlflow_model_name")
        with pytest.raises(ValueError, match="mlflow_model_name не задан"):
            merge_and_export.__wrapped__(base_cfg)

    def test_fail_get_production_version(
        self, base_cfg, mock_instantiate, mock_setup_config, mock_sys_exit
    ):
        """Если в MLflow нет Production модели, скрипт должен завершиться."""
        mock_logger = MagicMock()
        mock_logger.get_production_version.side_effect = Exception("No Production alias")
        mock_instantiate.return_value = mock_logger

        merge_and_export.__wrapped__(base_cfg)

        mock_sys_exit.assert_called_once_with(1)

    def test_skip_merge_if_already_exists(
        self, base_cfg, mock_instantiate, mock_setup_config, mock_peft, mocker
    ):
        """Бизнес-логика: если монолит vX уже есть в storage, слияние пропускается, но манифест обновляется."""
        mock_logger = MagicMock()
        mock_storage = MagicMock()
        mock_router = MagicMock()

        mock_instantiate.side_effect = [mock_logger, mock_storage, mock_router]
        mock_logger.get_production_version.return_value = "2"
        mock_storage.exists.return_value = True  # Симулируем, что монолит уже есть

        mock_router.download_manifest.return_value = {}

        merge_and_export.__wrapped__(base_cfg)

        # Убеждаемся, что PeftModel вообще не вызывался (слияние пропущено)
        mock_peft.from_pretrained.assert_not_called()

        # Проверяем, что манифест все равно обновился корректно
        mock_storage.upload_file.assert_called_once()
        manifest_path = Path(mock_storage.upload_file.call_args.kwargs["local_path"])
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["sequence_pipeline"]["load_type"] == "full_model"
        assert (
            manifest["sequence_pipeline"]["model_uri"]
            == "s3://bucket/merged_models/MyModel_prod_v2"
        )

    def test_full_merge_and_export_flow(
        self, base_cfg, mock_instantiate, mock_setup_config, mock_resolver, mock_peft, mock_torch
    ):
        """Комплексный тест успешного слияния (Merge & Unload) и экспорта."""
        mock_logger = MagicMock()
        mock_storage = MagicMock()
        mock_router = MagicMock()
        mock_tokenizer_builder = MagicMock()
        mock_model_builder = MagicMock()

        # Настраиваем цепочку возвратов для hydra.utils.instantiate
        mock_instantiate.side_effect = [
            mock_logger,
            mock_storage,
            mock_router,
            mock_tokenizer_builder,
            mock_model_builder,
        ]

        mock_logger.get_production_version.return_value = "3"
        mock_storage.exists.return_value = False

        # Настраиваем резолвер
        resolver_instance = mock_resolver.return_value
        resolver_instance.resolve_and_patch.return_value = (None, "/mock/lora/path", None)

        # Настраиваем мок модели и токенизатора
        mock_tokenizer = mock_tokenizer_builder.build.return_value
        mock_tokenizer.pad_token_id = 42
        mock_base_model = mock_model_builder.build.return_value

        # Настраиваем PEFT
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_merged_model.generation_config.pad_token_id = -1

        # Манифест со старым конфигом lora
        mock_router.download_manifest.return_value = {
            "sequence_pipeline": {
                "load_type": "lora",
                "base_model_uri": "hf://base",
                "lora_uri": "s3://bucket/old_lora",
                "keep_me": True,
            }
        }

        # --- Вызов функции ---
        merge_and_export.__wrapped__(base_cfg)

        # 1. Проверки резолвинга и модификаторов
        resolver_instance.resolve_and_patch.assert_called_once_with(
            base_cfg,
            "s3://bucket/manifest.json",
            pipeline_name="sequence_pipeline",
            is_training=False,
        )
        assert base_cfg.model.builder.modifiers is None  # Убеждаемся, что модификаторы отключились

        # 2. Проверки слияния
        mock_peft.from_pretrained.assert_called_once_with(mock_base_model, "/mock/lora/path")
        mock_peft_model.merge_and_unload.assert_called_once()
        assert mock_merged_model.generation_config.pad_token_id == 42

        # 3. Проверки сохранения
        mock_merged_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

        mock_storage.upload.assert_called_once()
        assert (
            mock_storage.upload.call_args.kwargs["remote_path"] == "merged_models/MyModel_prod_v3"
        )

        # 4. Очистка памяти
        mock_torch.cuda.empty_cache.assert_called_once()

        # 5. Проверка нового манифеста
        mock_storage.upload_file.assert_called_once()
        manifest_path = mock_storage.upload_file.call_args.kwargs["local_path"]
        with open(manifest_path) as f:
            new_manifest = json.load(f)

        pipe_cfg = new_manifest["sequence_pipeline"]
        assert pipe_cfg["load_type"] == "full_model"
        assert pipe_cfg["model_uri"] == "s3://bucket/merged_models/MyModel_prod_v3"
        assert "base_model_uri" not in pipe_cfg
        assert "lora_uri" not in pipe_cfg
        assert pipe_cfg["keep_me"] is True  # Старые метаданные сохранились

    def test_resolver_returns_empty_lora_path(
        self, base_cfg, mock_instantiate, mock_setup_config, mock_resolver
    ):
        """Если манифест сломан и резолвер не вернул путь к lora, должна быть ошибка FileNotFoundError."""
        mock_instantiate.side_effect = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        mock_instantiate.return_value.exists.return_value = False

        mock_resolver.return_value.resolve_and_patch.return_value = (None, None, None)

        with pytest.raises(FileNotFoundError, match="Резолвер не вернул lora_path"):
            merge_and_export.__wrapped__(base_cfg)
