from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

# Укажи правильный путь импорта
from src.endpoints.train import run_universal_train


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def base_cfg():
    """Базовый конфиг, имитирующий структуру Hydra."""
    return OmegaConf.create(
        {
            "seed": 42,
            "resume_training": False,
            "system": {
                "logger": {"experiment_logger": {"_target_": "dummy_logger"}},
                "storage_router": {"_target_": "dummy_router"},
                "manifest": {"uri": "s3://manifest.json"},
                "paths": {
                    "model_dir": "/tmp/models",
                    "benchmark_cache_dir": "/tmp/bench",
                    "processed_data_dir": "/tmp/data",
                    "log_dir": "/tmp/logs",
                },
            },
            "model": {"use_manifest": True},
            "data": {"dataset": "dummy"},
            "training": {
                "accelerator": "cpu",
                "callbacks": {
                    "retrieval_eval": {"_target_": "eval_cb"},
                    "regular_cb": {"_target_": "reg_cb"},
                },
                "optimizer": "adam",
                "loss": "cross_entropy",
            },
        }
    )


@pytest.fixture
def mock_instantiate(mocker):
    """Мокает hydra.utils.instantiate."""
    return mocker.patch("src.endpoints.train.hydra.utils.instantiate")


@pytest.fixture
def mock_deps(mocker):
    """Мокает тяжелые зависимости."""
    mocker.patch("src.endpoints.train.pl.seed_everything")

    mock_torch = mocker.patch("src.endpoints.train.torch")
    mock_torch.cuda.is_available.return_value = False

    mock_resolver = mocker.patch("src.endpoints.train.ArtifactResolver")
    mock_loader = mocker.patch("src.endpoints.train.BenchmarkLoader")
    mock_datamodule = mocker.patch("src.endpoints.train.DataModule")

    return {
        "torch": mock_torch,
        "resolver": mock_resolver,
        "loader": mock_loader,
        "datamodule": mock_datamodule,
    }


@pytest.fixture
def build_module_fn():
    """Фабрика для возврата моковых моделей."""
    mock_module = MagicMock()
    mock_base = MagicMock()
    mock_tokenizer = MagicMock()

    def _build(cfg, logger):
        return mock_module, mock_base, mock_tokenizer

    # Добавляем ссылки на объекты, чтобы проверять их в тестах
    _build.mock_module = mock_module
    _build.mock_base = mock_base
    _build.mock_tokenizer = mock_tokenizer
    return _build


# ===========================================================================
# Тесты проверок и резолвинга
# ===========================================================================


class TestUniversalTrainChecks:
    def test_cuda_check_fails(self, base_cfg, build_module_fn, mock_deps):
        """Если accelerator='gpu', но CUDA нет, должна быть ошибка."""
        base_cfg.training.accelerator = "gpu"

        with pytest.raises(RuntimeError, match="accelerator='gpu', но CUDA недоступна"):
            run_universal_train(base_cfg, "test_pipe", build_module_fn)

    def test_resolver_called(self, base_cfg, build_module_fn, mock_deps, mock_instantiate):
        """Проверка работы резолвера, если use_manifest = True."""
        mock_logger = MagicMock()
        mock_router = MagicMock()
        mock_instantiate.side_effect = [mock_logger, mock_router, MagicMock()]

        run_universal_train(base_cfg, "test_pipe", build_module_fn)

        resolver_inst = mock_deps["resolver"].return_value
        resolver_inst.resolve_and_patch.assert_called_once_with(
            base_cfg, "s3://manifest.json", pipeline_name="test_pipe", is_training=True
        )

    def test_callbacks_instantiation_and_config_cleanup(
        self, base_cfg, build_module_fn, mock_deps, mock_instantiate
    ):
        """Проверка, что логгер прокидывается только в нужные коллбэки, а лишние ключи удаляются."""
        mock_logger = MagicMock()
        mock_instantiate.side_effect = [
            mock_logger,  # experiment_logger
            MagicMock(),  # storage_router
            MagicMock(),  # eval_cb (retrieval_eval)
            MagicMock(),  # reg_cb
            MagicMock(),  # trainer
        ]

        run_universal_train(base_cfg, "test_pipe", build_module_fn)

        # Проверяем, что в eval_cb передали experiment_logger
        instantiate_calls = mock_instantiate.call_args_list
        assert "experiment_logger" in instantiate_calls[2].kwargs
        assert "experiment_logger" not in instantiate_calls[3].kwargs

        # Проверяем, что ключи очищены из cfg.training
        assert "optimizer" not in base_cfg.training
        assert "callbacks" not in base_cfg.training


# ===========================================================================
# Тесты флоу обучения и завершения
# ===========================================================================


class TestUniversalTrainFlow:
    @pytest.fixture
    def setup_happy_path(self, base_cfg, build_module_fn, mock_deps, mock_instantiate):
        mock_logger = MagicMock()
        mock_logger.get_run_id.return_value = "run_123"

        mock_trainer = MagicMock()
        mock_trainer.tested = False
        mock_trainer.checkpoint_callback.best_model_path = "/tmp/best.ckpt"
        mock_trainer.checkpoint_callback.best_model_score = 0.85

        # Настраиваем инстанциацию так, чтобы последним возвращался trainer
        mock_instantiate.side_effect = lambda cfg, **kwargs: (
            mock_logger
            if "dummy_logger" in str(cfg)
            else MagicMock()
            if "dummy_router" in str(cfg)
            else mock_trainer
            if "accelerator" in cfg
            else MagicMock()
        )

        return mock_logger, mock_trainer

    def test_happy_path_fit_and_test(self, base_cfg, build_module_fn, mock_deps, setup_happy_path):
        """Успешный прогон: fit -> load_best_weights -> test -> save_adapter."""
        mock_logger, mock_trainer = setup_happy_path

        # Эмулируем, что базовая модель это PeftModel
        import peft

        build_module_fn.mock_base.__class__ = peft.PeftModel

        run_universal_train(base_cfg, "test_pipe", build_module_fn)

        # Проверяем запуск обучения
        mock_trainer.fit.assert_called_once()

        # Проверяем загрузку чекпоинта и запуск теста
        mock_deps["torch"].load.assert_called_once_with(
            "/tmp/best.ckpt", map_location=build_module_fn.mock_module.device, weights_only=False
        )
        mock_trainer.test.assert_called_once()

        # Проверяем сохранение адаптера
        mock_logger.save_adapter.assert_called_once_with(
            cfg=base_cfg,
            model_module=build_module_fn.mock_module,
            tokenizer=build_module_fn.mock_tokenizer,
            run_id="run_123",
            best_score=0.85,
            pipeline_name="test_pipe",
        )

    def test_keyboard_interrupt_graceful_exit(
        self, base_cfg, build_module_fn, mock_deps, setup_happy_path
    ):
        """Прерывание Ctrl+C должно штатно переходить к тесту и сохранению."""
        mock_logger, mock_trainer = setup_happy_path

        # Заставляем fit выбросить прерывание
        mock_trainer.fit.side_effect = KeyboardInterrupt()

        # Скрипт не должен упасть
        run_universal_train(base_cfg, "test_pipe", build_module_fn)

        # Но тест должен быть вызван
        mock_trainer.test.assert_called_once()

    def test_resume_training_loads_last_ckpt(
        self, base_cfg, build_module_fn, mock_deps, setup_happy_path, mocker
    ):
        """Если resume_training=True и есть last.ckpt, путь прокидывается в fit."""
        _, mock_trainer = setup_happy_path
        base_cfg.resume_training = True

        # Мокаем проверку существования файла last.ckpt
        mocker.patch("pathlib.Path.exists", return_value=True)

        run_universal_train(base_cfg, "test_pipe", build_module_fn)

        # Убеждаемся, что ckpt_path передан
        fit_kwargs = mock_trainer.fit.call_args.kwargs
        assert "last.ckpt" in fit_kwargs["ckpt_path"]
