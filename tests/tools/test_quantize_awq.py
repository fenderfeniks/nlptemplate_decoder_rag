import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

# Укажи правильный путь импорта
from src.tools.quantize_awq import _load_calibration_data, quantize_and_export


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def base_cfg():
    """Базовый конфиг для тестов."""
    return OmegaConf.create(
        {
            "pipeline_name": "test_pipeline",
            "model": {
                "architecture": {"mlflow_model_name": "TestModel"},
                "tokenizer": {"_target_": "dummy_tokenizer"},
                "builder": {"model_name_or_path": "/tmp/local/model/path"},
            },
            "awq": {
                "w_bit": 4,
                "q_group_size": 128,
                "zero_point": True,
                "version": "GEMM",
                "n_calib_samples": 128,
            },
            "data": {"query_column": "question"},
            "system": {
                "storage": {"_target_": "dummy", "uri_prefix": "s3://bucket/"},
                "storage_router": {"_target_": "dummy"},
                "manifest": {"uri": "s3://bucket/manifest.json"},
                "paths": {"model_dir": "/tmp/models"},
            },
        }
    )


@pytest.fixture
def mock_sys_exit(mocker):
    return mocker.patch("src.tools.quantize_awq.sys.exit")


@pytest.fixture
def mock_cuda(mocker):
    """Принудительно говорим, что CUDA доступна, чтобы тесты не падали на CPU-машинах."""
    mock = mocker.patch("src.tools.quantize_awq.torch.cuda.is_available")
    mock.return_value = True
    return mock


@pytest.fixture
def mock_awq(mocker):
    """Мокаем библиотеку awq, чтобы тесты проходили без ее физической установки."""
    mock_awq_module = MagicMock()
    mock_auto_awq = MagicMock()
    mock_awq_module.AutoAWQForCausalLM = mock_auto_awq
    mocker.patch.dict("sys.modules", {"awq": mock_awq_module})
    return mock_auto_awq


@pytest.fixture
def mock_instantiate(mocker):
    return mocker.patch("src.tools.quantize_awq.hydra.utils.instantiate")


@pytest.fixture
def mock_setup_config(mocker):
    mock = mocker.patch("src.tools.quantize_awq.setup_config")
    mock.side_effect = lambda x: x
    return mock


# ===========================================================================
# Тесты утилит (Загрузка калибровочных данных)
# ===========================================================================


class TestLoadCalibrationData:
    @pytest.fixture
    def mock_loader(self, mocker):
        return mocker.patch("src.tools.quantize_awq.BenchmarkLoader")

    def test_returns_empty_when_no_dataset(self, base_cfg, mock_loader):
        """Если датасет не загрузился (None) или пустой, возвращаем пустой список."""
        instance = mock_loader.return_value
        instance.load_as_dataset.return_value = None

        result = _load_calibration_data(base_cfg, MagicMock(), Path("/tmp"))
        assert result == []

    def test_extracts_correct_text_column(self, base_cfg, mock_loader):
        """Проверяем, что берется правильная колонка и ограничивается по n_samples."""
        instance = mock_loader.return_value

        # Симулируем HuggingFace Dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["id", "question", "answer"]
        # Делаем датасет итерируемым
        data_rows = [{"id": i, "question": f"Q{i}"} for i in range(10)]
        mock_dataset.__iter__.return_value = iter(data_rows)
        mock_dataset.__len__.return_value = 10

        instance.load_as_dataset.return_value = mock_dataset

        # Запрашиваем 5 семплов из колонки question
        result = _load_calibration_data(
            base_cfg, MagicMock(), Path("/tmp"), query_column="question", n_samples=5
        )

        assert len(result) == 5
        assert result == ["Q0", "Q1", "Q2", "Q3", "Q4"]

    def test_fallback_column_search(self, base_cfg, mock_loader):
        """Если query_column не задан, код должен найти колонку-фоллбэк (например, 'text')."""
        instance = mock_loader.return_value
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["meta", "text"]
        mock_dataset.__iter__.return_value = iter([{"text": "hello"}])
        mock_dataset.__len__.return_value = 1

        instance.load_as_dataset.return_value = mock_dataset

        # Не передаем query_column
        result = _load_calibration_data(
            base_cfg, MagicMock(), Path("/tmp"), query_column=None, n_samples=1
        )
        assert result == ["hello"]


# ===========================================================================
# Тесты основного пайплайна квантизации
# ===========================================================================


class TestQuantizeAndExport:
    def test_no_cuda_exits(self, base_cfg, mock_cuda, mock_sys_exit, mock_setup_config):
        """Если CUDA нет, скрипт обязан прерваться."""
        mock_cuda.return_value = False
        quantize_and_export.__wrapped__(base_cfg)
        mock_sys_exit.assert_called_once_with(1)

    def test_missing_awq_exits(self, base_cfg, mock_cuda, mock_sys_exit, mock_setup_config):
        """Если пакет awq не установлен, скрипт должен упасть (имитация без мока sys.modules)."""
        # Гарантируем, что awq выбросит ImportError
        with patch.dict("sys.modules", {"awq": None}):
            quantize_and_export.__wrapped__(base_cfg)
            mock_sys_exit.assert_called_once_with(1)

    def test_skip_quantization_if_exists(
        self, base_cfg, mock_cuda, mock_awq, mock_instantiate, mock_setup_config
    ):
        """Если модель уже квантизована, пропускаем AutoAWQ и просто обновляем манифест."""
        mock_storage = MagicMock()
        mock_router = MagicMock()
        mock_instantiate.side_effect = [mock_storage, mock_router]

        mock_storage.exists.return_value = True
        mock_router.download_manifest.return_value = {}

        quantize_and_export.__wrapped__(base_cfg)

        # Проверяем, что модель не скачивалась и квантизация не запускалась
        mock_awq.from_pretrained.assert_not_called()

        # Но манифест был обновлен
        mock_storage.upload_file.assert_called_once()
        manifest_path = Path(mock_storage.upload_file.call_args.kwargs["local_path"])
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["test_pipeline"]["load_type"] == "full_model"
        assert manifest["test_pipeline"]["model_uri"] == "s3://bucket/awq_models/TestModel_w4g128"

    @patch("src.tools.quantize_awq.ArtifactResolver")
    def test_lora_path_in_resolver_exits(
        self,
        mock_resolver_cls,
        base_cfg,
        mock_cuda,
        mock_awq,
        mock_instantiate,
        mock_setup_config,
        mock_sys_exit,
    ):
        """Бизнес-логика: если резолвер вернул lora_path, квантизация запрещена (нужен merge)."""
        mock_storage = MagicMock()
        mock_instantiate.side_effect = [mock_storage, MagicMock()]
        mock_storage.exists.return_value = False

        mock_resolver = mock_resolver_cls.return_value
        # Возвращаем lora_path != None
        mock_resolver.resolve_and_patch.return_value = (None, "/path/to/lora", None)

        quantize_and_export.__wrapped__(base_cfg)

        mock_sys_exit.assert_called_once_with(1)

    @patch("src.tools.quantize_awq.ArtifactResolver")
    @patch("src.tools.quantize_awq._load_calibration_data")
    @patch("src.tools.quantize_awq.torch.cuda.empty_cache")
    def test_happy_path_quantization(
        self,
        mock_empty_cache,
        mock_load_data,
        mock_resolver_cls,
        base_cfg,
        mock_cuda,
        mock_awq,
        mock_instantiate,
        mock_setup_config,
    ):
        """Полный флоу успешной квантизации: резолвинг -> загрузка -> AWQ -> выгрузка -> манифест."""
        mock_storage = MagicMock()
        mock_router = MagicMock()
        mock_tokenizer_builder = MagicMock()

        # instantiate вызывается 3 раза: storage, router, tokenizer (в блоке 4)
        mock_instantiate.side_effect = [mock_storage, mock_router, mock_tokenizer_builder]
        mock_storage.exists.return_value = False

        # Резолвер возвращает None для lora_path
        mock_resolver_cls.return_value.resolve_and_patch.return_value = (None, None, None)

        mock_load_data.return_value = ["calib1", "calib2"]
        mock_router.download_manifest.return_value = {
            "test_pipeline": {"lora_uri": "s3://old", "keep_meta": "yes"}
        }

        # Мокируем AutoAWQ
        mock_model = mock_awq.from_pretrained.return_value

        # === ВЫПОЛНЕНИЕ ===
        quantize_and_export.__wrapped__(base_cfg)

        # 1. Проверяем вызов AutoAWQ
        mock_awq.from_pretrained.assert_called_once_with("/tmp/local/model/path", device_map="auto")
        mock_model.quantize.assert_called_once()
        quant_kwargs = mock_model.quantize.call_args.kwargs
        assert quant_kwargs["calib_data"] == ["calib1", "calib2"]
        assert quant_kwargs["quant_config"]["w_bit"] == 4

        # 2. Проверяем локальное сохранение
        mock_model.save_quantized.assert_called_once()

        # 3. Проверяем выгрузку в Storage
        mock_storage.upload.assert_called_once()
        upload_kwargs = mock_storage.upload.call_args.kwargs
        assert upload_kwargs["remote_path"] == "awq_models/TestModel_w4g128"

        # 4. Проверяем очистку памяти
        mock_empty_cache.assert_called_once()

        # 5. Проверяем правильность нового манифеста
        mock_storage.upload_file.assert_called_once()
        manifest_path = mock_storage.upload_file.call_args.kwargs["local_path"]
        with open(manifest_path) as f:
            new_manifest = json.load(f)

        pipe_cfg = new_manifest["test_pipeline"]
        assert pipe_cfg["load_type"] == "full_model"
        assert pipe_cfg["model_uri"] == "s3://bucket/awq_models/TestModel_w4g128"
        assert pipe_cfg["quantization"]["method"] == "awq"
        assert pipe_cfg["quantization"]["w_bit"] == 4
        assert "lora_uri" not in pipe_cfg  # Удалилось
        assert pipe_cfg["keep_meta"] == "yes"  # Старое не затерлось
