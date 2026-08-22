from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.utils.logging.mlflow_logger import (
    LightningMLflowLogger,
    MLflowLogger,
    _extract_run_id_from_trainer,
    _find_adapter_config,
)


# ===========================================================================
# Фикстуры для изоляции тяжелых зависимостей
# ===========================================================================


@pytest.fixture
def mock_mlflow(mocker):
    """Мокает глобальный объект mlflow, используемый в mlflow_logger."""
    return mocker.patch("src.utils.logging.mlflow_logger.mlflow")


@pytest.fixture
def mock_mlflow_client_class(mocker):
    """Мокает класс MlflowClient."""
    return mocker.patch("src.utils.logging.mlflow_logger.MlflowClient")


@pytest.fixture
def mock_register_globals(mocker):
    """Мокает функцию из src.utils.torch_utils."""
    return mocker.patch("src.utils.logging.mlflow_logger.register_safe_globals")


@pytest.fixture
def base_config():
    """Базовый Hydra-конфиг для тестов."""
    return OmegaConf.create(
        {
            "model": {"architecture": {"mlflow_model_name": "TestModel"}},
            "logger": {
                "registry": {
                    "artifact_path": "lora_weights",
                    "register_on_success": True,
                    "promote_to_staging": True,
                }
            },
        }
    )


# ===========================================================================
# Тесты приватных утилит
# ===========================================================================


def test_extract_run_id_from_trainer(mocker):
    trainer = MagicMock()
    trainer.logger.run_id = "test_run_123"
    assert _extract_run_id_from_trainer(trainer) == "test_run_123"

    trainer.logger = None
    mock_active = mocker.patch("src.utils.logging.mlflow_logger.mlflow.active_run")
    mock_active.return_value.info.run_id = "active_run_456"
    assert _extract_run_id_from_trainer(trainer) == "active_run_456"


def test_find_adapter_config(tmp_path):
    # Симуляция скачанного артифакта без конфига
    assert _find_adapter_config(tmp_path) is None

    # Симуляция скачанного артифакта с конфигом в папке lora_weights
    target_dir = tmp_path / "lora_weights"
    target_dir.mkdir()
    (target_dir / "adapter_config.json").touch()

    found_path = _find_adapter_config(tmp_path)
    assert found_path == target_dir


# ===========================================================================
# Тесты MLflowLogger (Standalone)
# ===========================================================================


class TestMLflowLogger:
    def test_init_creates_experiment(self, mock_mlflow):
        """Проверка бизнес-логики инициализации: если эксперимента нет, он создается."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new_exp_id"

        MLflowLogger(experiment_name="NewExp", artifact_location="file:///tmp")

        mock_mlflow.create_experiment.assert_called_once_with(
            name="NewExp", artifact_location="file:///tmp"
        )
        mock_mlflow.set_experiment.assert_called_once_with(experiment_id="new_exp_id")

    def test_log_metrics(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="TestExp")
        mock_mlflow.active_run.return_value = True

        logger.log_metrics({"loss": 0.5, "acc": 0.9}, stage="val", step=10)

        mock_mlflow.log_metric.assert_any_call("val_loss", 0.5, step=10)
        mock_mlflow.log_metric.assert_any_call("val_acc", 0.9, step=10)

    def test_save_adapter_success(
        self, mock_mlflow, mock_mlflow_client_class, mock_register_globals, base_config
    ):
        """Проверка флоу сохранения адаптера с мокированием сохранения весов модели."""
        logger = MLflowLogger()
        client_instance = mock_mlflow_client_class.return_value
        client_instance.create_model_version.return_value.version = "1"

        model_module = MagicMock()
        tokenizer = MagicMock()

        # Правильный мок save_pretrained: он должен создавать adapter_config.json
        # иначе _save_adapter_to_tempdir выбросит FileNotFoundError
        def mock_save_pretrained(path):
            (Path(path) / "adapter_config.json").touch()

        model_module.model.save_pretrained.side_effect = mock_save_pretrained

        logger.save_adapter(
            cfg=base_config,
            model_module=model_module,
            tokenizer=tokenizer,
            run_id="run_123",
            pipeline_name="train",
            best_score=0.15,
        )

        # Проверки бизнес-логики Registry
        mock_mlflow.log_artifacts.assert_called_once()
        client_instance.create_registered_model.assert_called_with("TestModel_LoRA")
        client_instance.create_model_version.assert_called_once()
        client_instance.set_registered_model_alias.assert_called_with(
            name="TestModel_LoRA", alias="Staging", version="1"
        )
        client_instance.set_model_version_tag.assert_called_with(
            "TestModel_LoRA", "1", "val_loss", "0.15"
        )

    def test_load_adapter_by_run_id(self, mock_mlflow, tmp_path):
        """Проверка загрузки адаптера по run_id."""
        logger = MLflowLogger()
        resume_cfg = {"enabled": True, "run_id": "run_123", "artifact_path": "lora_weights"}

        # Мокаем скачивание, возвращая временную директорию с фейковым конфигом
        fake_download_path = tmp_path / "downloaded"
        target_dir = fake_download_path / "lora_weights"
        target_dir.mkdir(parents=True)
        (target_dir / "adapter_config.json").touch()

        mock_mlflow.artifacts.download_artifacts.return_value = str(fake_download_path)

        loaded_path = logger.load_adapter(resume_cfg)

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_123", artifact_path="lora_weights"
        )
        assert loaded_path == str(target_dir)

    def test_promote_model_success(self, mock_mlflow_client_class):
        """Бизнес-логика: Staging лучше Production -> промоут."""
        logger = MLflowLogger()
        client_instance = mock_mlflow_client_class.return_value

        staging_mv = MagicMock(version="2", tags={"val_loss": "0.10"})
        prod_mv = MagicMock(version="1", tags={"val_loss": "0.20"})

        client_instance.get_model_version_by_alias.side_effect = [staging_mv, prod_mv]

        result = logger.promote_model("TestModel")

        assert result is True
        client_instance.set_registered_model_alias.assert_called_once_with(
            "TestModel", "Production", "2"
        )

    def test_promote_model_reject(self, mock_mlflow_client_class):
        """Бизнес-логика: Staging хуже Production -> отказ в промоуте."""
        logger = MLflowLogger()
        client_instance = mock_mlflow_client_class.return_value

        staging_mv = MagicMock(version="3", tags={"val_loss": "0.50"})
        prod_mv = MagicMock(version="2", tags={"val_loss": "0.20"})

        client_instance.get_model_version_by_alias.side_effect = [staging_mv, prod_mv]

        result = logger.promote_model("TestModel")

        assert result is False
        client_instance.set_registered_model_alias.assert_not_called()


# ===========================================================================
# Тесты LightningMLflowLogger
# ===========================================================================


class TestLightningMLflowLogger:
    def test_log_metrics_delegates_to_pl_module(self):
        trainer = MagicMock()
        pl_module = MagicMock()
        logger = LightningMLflowLogger(trainer, pl_module)

        metrics = {"loss": 0.5}
        logger.log_metrics(metrics, stage="train", step=1)

        # У Lightning метрики пишутся напрямую в pl_module.log с нужными флагами DDP
        pl_module.log.assert_called_once_with(
            "train_loss", 0.5, sync_dist=True, prog_bar=True, logger=True
        )

    def test_log_table_with_trainer_logger(self, mocker):
        trainer = MagicMock()
        pl_module = MagicMock()
        trainer.logger.run_id = "trainer_run_123"

        logger = LightningMLflowLogger(trainer, pl_module)
        df = pd.DataFrame({"col1": [1, 2]})

        logger.log_table(df, stage="val", step=5)

        # Проверяем, что используется метод .experiment у трейнера, если он есть
        trainer.logger.experiment.log_table.assert_called_once_with(
            run_id="trainer_run_123", data=df, artifact_file="generations/val_step_5_results.json"
        )
