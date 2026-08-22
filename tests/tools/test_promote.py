import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

# Укажи правильный путь импорта в зависимости от структуры проекта
from src.tools.promote import main


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def base_cfg():
    """Генерирует базовый конфиг Hydra без хардкода внутри самих тестов."""
    return OmegaConf.create(
        {
            "pipeline_name": "nlp_pipeline",
            "model": {
                "architecture": {
                    "mlflow_model_name": "MyModel",
                    "model_name_or_path": "some-org/my-model",
                    "base_model_uri": "hf://some-org/my-model",
                }
            },
            "system": {
                "logger": {
                    "experiment_logger": {"_target_": "dummy"},
                    "registry": {"artifact_path": "lora_weights"},
                },
                "storage": {"_target_": "dummy", "uri_prefix": "s3://bucket/"},
                "storage_router": {"_target_": "dummy"},
                "manifest": {"uri": "s3://bucket/manifest.json"},
            },
        }
    )


@pytest.fixture
def mock_instantiate(mocker):
    """Мокает инстанциацию классов через Hydra."""
    return mocker.patch("src.tools.promote.hydra.utils.instantiate")


@pytest.fixture
def mock_sys_exit(mocker):
    """Мокает sys.exit, чтобы тесты не прерывались."""
    return mocker.patch("src.tools.promote.sys.exit")


@pytest.fixture
def mock_setup_config(mocker):
    """Мокает утилиту настройки конфига, возвращая переданный конфиг."""
    mock = mocker.patch("src.tools.promote.setup_config")
    mock.side_effect = lambda x: x
    return mock


# ===========================================================================
# Тесты бизнес-логики и краевых случаев
# ===========================================================================


class TestPromoteScript:
    def test_missing_mlflow_model_name_raises_error(self, base_cfg, mock_setup_config):
        """Валидация конфигурации: ошибка при отсутствии mlflow_model_name."""
        base_cfg.model.architecture.pop("mlflow_model_name")

        with pytest.raises(ValueError, match="mlflow_model_name не задан"):
            # Вызываем распакованную функцию для обхода @hydra.main
            main.__wrapped__(base_cfg)

    def test_promote_model_exception_triggers_exit(
        self, base_cfg, mock_instantiate, mock_sys_exit, mock_setup_config
    ):
        """Бизнес-логика: если логгер не смог продвинуть модель, скрипт прерывается."""
        mock_logger = MagicMock()
        # Возвращаем mock_logger на первый вызов instantiate
        mock_instantiate.side_effect = [mock_logger, MagicMock(), MagicMock()]
        mock_logger.promote_model.side_effect = Exception("Registry connection failed")

        main.__wrapped__(base_cfg)

        mock_sys_exit.assert_called_once_with(1)

    def test_load_adapter_fails_triggers_exit(
        self, base_cfg, mock_instantiate, mock_sys_exit, mock_setup_config
    ):
        """Бизнес-логика: если не удалось загрузить адаптер, скрипт прерывается."""
        mock_logger = MagicMock()
        mock_router = MagicMock()

        mock_instantiate.side_effect = [mock_logger, MagicMock(), mock_router]
        mock_router.download_manifest.return_value = {}
        # Эмулируем ситуацию, когда адаптер не найден
        mock_logger.load_adapter.return_value = None

        main.__wrapped__(base_cfg)

        mock_sys_exit.assert_called_once_with(1)
        mock_logger.load_adapter.assert_called_once()

    def test_successful_promotion_and_manifest_update(
        self, base_cfg, mock_instantiate, mock_setup_config, mocker
    ):
        """
        Комплексный тест счастливого пути:
        1. Успешный промоут модели.
        2. Скачивание старого манифеста и корректное разрешение base_model_uri.
        3. Загрузка адаптера в хранилище.
        4. Обновление манифеста без затирания соседних пайплайнов.
        """
        mock_logger = MagicMock()
        mock_storage = MagicMock()
        mock_router = MagicMock()

        # Настраиваем возвращаемые значения для трех вызовов instantiate
        mock_instantiate.side_effect = [mock_logger, mock_storage, mock_router]
        mock_logger.load_adapter.return_value = "/local/tmp/path/to/adapter"

        # Имитируем существующий манифест с другим пайплайном
        existing_manifest = {
            "nlp_pipeline": {"model_uri": "hf://old-model", "other_key": "should_be_kept"},
            "vision_pipeline": {"load_type": "full"},
        }
        mock_router.download_manifest.return_value = existing_manifest

        main.__wrapped__(base_cfg)

        # 1. Проверяем обращение к логгеру
        mock_logger.promote_model.assert_called_once_with(
            reg_model_name="MyModel_LoRA",
            staging_alias="Staging",
            production_alias="Production",
            metric_tag="val_loss",
        )

        # 2. Проверяем загрузку адаптера в хранилище
        mock_storage.upload.assert_called_once_with(
            local_dir="/local/tmp/path/to/adapter", remote_path="adapters/MyModel_prod"
        )

        # 3. Проверяем параметры, переданные в upload_file (загрузка манифеста)
        mock_storage.upload_file.assert_called_once()
        upload_kwargs = mock_storage.upload_file.call_args.kwargs
        assert upload_kwargs["remote_path"] == "manifest.json"

        # Читаем локально сохраненный манифест, чтобы проверить структуру
        manifest_path = Path(upload_kwargs["local_path"])
        with open(manifest_path, encoding="utf-8") as f:
            new_manifest = json.load(f)

        # Убеждаемся, что другой пайплайн остался нетронутым
        assert "vision_pipeline" in new_manifest

        # Убеждаемся, что текущий пайплайн обновился корректно
        updated_nlp = new_manifest["nlp_pipeline"]
        assert updated_nlp["load_type"] == "lora"
        assert updated_nlp["base_model_uri"] == "hf://old-model"  # Вытянуло из model_uri
        assert updated_nlp["lora_uri"] == "s3://bucket/adapters/MyModel_prod"
        assert "model_uri" not in updated_nlp  # Попнули старый ключ
        assert "updated_at" in updated_nlp
        assert updated_nlp["other_key"] == "should_be_kept"  # Старые ключи не затерты
