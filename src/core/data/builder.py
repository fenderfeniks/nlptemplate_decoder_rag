# src/core/data/builder.py
"""
Модуль управления данных для NLP пайплайна.

Содержит реализацию PyTorch Lightning DataModule, который инкапсулирует
всю логику загрузки, очистки, кэширования и подготовки батчей текста.
Поддерживает динамическую токенизацию и версионирование кэша на основе
хэширования полной конфигурации обработки данных.
"""

import logging
import os
import json
import hashlib
from typing import Any, Optional

import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate
from datasets import load_from_disk, DatasetDict
from transformers import PreTrainedTokenizerBase

from src.core.data.datasets import NLPDatasetAdapter

logger = logging.getLogger(__name__)

class NLPDataModule(pl.LightningDataModule):
    """
    Lightning DataModule для обработки текстовых данных.
    """

    def __init__(self, data_cfg: Any, tokenizer: PreTrainedTokenizerBase):
        """
        Инициализирует DataModule и вычисляет путь для DVC кэша.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        
        hash_dict = {
            "cleaner": OmegaConf.to_container(self.data_cfg.cleaner, resolve=True),
            "source": OmegaConf.to_container(self.data_cfg.source, resolve=True),
            "text_column": self.data_cfg.get("text_column"),
            "target_column": self.data_cfg.get("target_column"),
            "seed": self.data_cfg.get("seed"),
            "max_length": self.data_cfg.get("max_length")
        }
        
        hash_str = json.dumps(hash_dict, sort_keys=True)
        config_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()[:8]
        
        dataset_name = self.data_cfg.get("dataset_name", "nlp_dataset")
        
        self.processed_dir = os.path.join(
            self.data_cfg.paths.processed_data_dir, 
            f"{dataset_name}_cleaned_{config_hash}"
        )

    def prepare_data(self) -> None:
        """
        Скачивает сырые данные, применяет очистку и сохраняет результат на диск.
        """
        if os.path.exists(self.processed_dir) and not self.data_cfg.get("force_reprocess", False):
            logger.info(f"Нашли кэш данных: {self.processed_dir}. Очистка пропущена.")
            return

        logger.info("Начинаем загрузку и обработку сырых данных...")
        
        raw_datasets = instantiate(self.data_cfg.source)
        
        # Разделение на train/val/test
        if "validation" in raw_datasets and "test" in raw_datasets:
            raw_train = raw_datasets["train"]
            raw_val = raw_datasets["validation"]
            raw_test = raw_datasets["test"]
        else:
            # Fallback: Если сплитов нет, делаем их искусственно из train
            split_ds = raw_datasets["train"].train_test_split(
                test_size=self.data_cfg.val_split_size * 2, # Берем x2, чтобы поделить на val и test
                seed=self.data_cfg.seed
            )
            raw_train = split_ds["train"]
            
            # Делим отщипнутый кусок пополам между val и test
            val_test_split = split_ds["test"].train_test_split(test_size=0.5, seed=self.data_cfg.seed)
            raw_val = val_test_split["train"]
            raw_test = val_test_split["test"]

        cleaner_pipeline = instantiate(self.data_cfg.cleaner)

        # Функция для минимизации дублирования кода
        def _process_split(dataset_split):
            return NLPDatasetAdapter(
                hf_dataset=dataset_split, 
                text_column=self.data_cfg.text_column, 
                cleaning_pipeline=cleaner_pipeline,
                num_proc=self.data_cfg.get("preprocessing_num_workers", 4),
                batch_size=self.data_cfg.get("preprocessing_batch_size", 1000)
            ).prepare_dataset()

        processed_dataset = DatasetDict({
            "train": _process_split(raw_train),
            "validation": _process_split(raw_val),
            "test": _process_split(raw_test)
        })
        
        processed_dataset.save_to_disk(self.processed_dir)
        logger.info(f"Данные успешно очищены и сохранены в {self.processed_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Загружает обработанные данные с диска в память текущего процесса.
        """
        processed_dataset = load_from_disk(self.processed_dir)
        
        if stage == "fit" or stage is None:
            self.train_dataset = processed_dataset["train"]
            self.val_dataset = processed_dataset["validation"]
            
        if stage == "test" or stage is None:
            self.test_dataset = processed_dataset["test"]

        self.collator = instantiate(
            self.data_cfg.collator, 
            tokenizer=self.tokenizer
        )

    def train_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.train_dataset,
            collate_fn=self.collator,
            shuffle=True
        )

    def val_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.val_dataset,
            collate_fn=self.collator,
            shuffle=False
        )
        
    def test_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.test_dataset,
            collate_fn=self.collator,
            shuffle=False
        )