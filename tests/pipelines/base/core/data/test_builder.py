# tests/pipelines/base/core/data/test_builder.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from src.pipelines.base.core.data.builder import DataModule, _TOKENIZATION_MARKER


@pytest.fixture
def dummy_tokenizer():
    """Фикстура для эмуляции токенизатора."""
    tokenizer = MagicMock()
    tokenizer.name_or_path = "test-tokenizer"
    return tokenizer


@pytest.fixture
def base_data_cfg(tmp_path):
    """Базовая конфигурация OmegaConf для DataModule."""
    return OmegaConf.create({
        "dataset_name": "test_ds",
        "seed": 42,
        "paths": {
            "processed_data_dir": str(tmp_path / "processed")
        },
        "source": {"_target_": "dummy.Source"},
        "splitter": {"_target_": "dummy.Splitter"},
        "transforms": [],
        "collator": {"_target_": "dummy.Collator"},
        "dataloader": {
            "batch_size": 16,
            "num_workers": 2,
            "shuffle": True, # Должно быть удалено в _dataloader_kwargs
            "drop_last": False
        }
    })


class TestDataModuleInitAndHash:
    def test_resolve_processed_dir_consistency(self, base_data_cfg, dummy_tokenizer):
        """Проверка, что одинаковый конфиг дает одинаковый хэш и путь."""
        dm1 = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        dm2 = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        
        assert dm1.processed_dir == dm2.processed_dir
        assert "test_ds_processed_" in str(dm1.processed_dir)

    def test_resolve_processed_dir_changes_with_config(self, base_data_cfg, dummy_tokenizer):
        """Если конфиг меняется, кэш-директория тоже должна измениться."""
        dm1 = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        
        base_data_cfg.seed = 999
        dm2 = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        
        assert dm1.processed_dir != dm2.processed_dir


class TestDataModuleTransforms:
    @patch("src.pipelines.base.core.data.builder.instantiate")
    def test_build_transforms_invalid_type(self, dummy_tokenizer):
        """Если transforms не может быть приведен к списку, должна быть ошибка."""
        # Используем MagicMock вместо OmegaConf для безопасного падения теста
        data_cfg = MagicMock()
        data_cfg.transforms = "не_список_и_не_словарь"
        
        dm = DataModule(data_cfg=data_cfg, tokenizer=dummy_tokenizer)
        with pytest.raises(TypeError, match="должен быть списком"):
            dm._build_transforms()

    def test_build_transforms_invalid_type(self, base_data_cfg, dummy_tokenizer):
        """Если transforms не может быть приведен к списку, должна быть ошибка."""
        # 1. Создаем объект с правильным конфигом, чтобы __init__ отработал успешно
        dm = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        
        # 2. Подменяем данные на невалидные уже внутри готового объекта
        dm.data_cfg.transforms = "не_список_и_не_словарь"
        
        # 3. Вызываем целевой метод и ждем наш TypeError
        with pytest.raises(TypeError, match="должен быть списком"):
            dm._build_transforms()


class TestDataModuleSubsample:
    def test_maybe_subsample_no_limit(self, base_data_cfg, dummy_tokenizer):
        dm = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        ds = Dataset.from_dict({"a": [1, 2, 3]})
        
        result = dm._maybe_subsample(ds, "train")
        assert len(result) == 3

    def test_maybe_subsample_float_ratio(self, base_data_cfg, dummy_tokenizer):
        """Проверка сабсемплинга по доле."""
        base_data_cfg.max_samples = 0.5
        dm = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        ds = Dataset.from_dict({"a": [1, 2, 3, 4]})
        
        result = dm._maybe_subsample(ds, "train")
        assert len(result) == 2

    def test_maybe_subsample_int_count(self, base_data_cfg, dummy_tokenizer):
        """Проверка сабсемплинга по абсолютному количеству."""
        base_data_cfg.max_samples = 2
        dm = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        ds = Dataset.from_dict({"a": [1, 2, 3, 4, 5]})
        
        result = dm._maybe_subsample(ds, "train")
        assert len(result) == 2


class TestDataModulePipeline:
    @patch("src.pipelines.base.core.data.builder.DatasetDict.save_to_disk")
    @patch("src.pipelines.base.core.data.builder.instantiate")
    def test_prepare_data_full_run(self, mock_instantiate, mock_save, base_data_cfg, dummy_tokenizer):
        """Тест полного пайплайна: загрузка -> сплит -> трансформ -> сохранение."""
        dm = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        
        # Мокаем источник (fetcher) и сплиттер
        mock_fetcher = MagicMock()
        mock_splitter = MagicMock()
        
        # Сплиттер возвращает фейковый датасет
        dummy_ds = Dataset.from_dict({"val": [1, 2]})
        mock_splitter.return_value = {"train": dummy_ds, "validation": dummy_ds}
        
        # instantiate вызывается 2 раза: для source и для splitter
        mock_instantiate.side_effect = [mock_fetcher, mock_splitter]
        
        dm.prepare_data()
        
        mock_fetcher.load.assert_called_once()
        mock_splitter.assert_called_once()
        mock_save.assert_called_once_with(str(dm.processed_dir))

    @patch("src.pipelines.base.core.data.builder.instantiate")
    def test_prepare_data_cache_hit(self, mock_instantiate, base_data_cfg, dummy_tokenizer):
        """Если кэш есть и force_reprocess=False, пайплайн пропускается."""
        dm = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        dm.processed_dir.mkdir(parents=True) # Эмулируем наличие кэша
        
        dm.prepare_data()
        
        mock_instantiate.assert_not_called()

    @patch("src.pipelines.base.core.data.builder.load_from_disk")
    @patch("src.pipelines.base.core.data.builder.instantiate")
    def test_setup_and_dataloaders(self, mock_instantiate, mock_load, base_data_cfg, dummy_tokenizer):
        """Проверка загрузки сплитов и генерации DataLoader-ов."""
        dm = DataModule(data_cfg=base_data_cfg, tokenizer=dummy_tokenizer)
        
        dummy_ds = Dataset.from_dict({"a": [1]})
        mock_load.return_value = {"train": dummy_ds, "test": dummy_ds}
        mock_instantiate.return_value = MagicMock() # коллатор
        
        dm.setup()
        
        assert dm.train_dataset is dummy_ds
        assert dm.val_dataset is None  # В моке нет validation
        assert dm.test_dataset is dummy_ds
        
        # Проверяем dataloader
        train_dl = dm.train_dataloader()
        assert train_dl.batch_size == 16
        
        # val_dataloader должен вернуть None, так как val_dataset = None
        assert dm.val_dataloader() is None
        
        test_dl = dm.test_dataloader()
        assert test_dl is not None