"""
Модуль управления данными для NLP пайплайна.

Содержит реализацию PyTorch Lightning DataModule, который инкапсулирует
всю логику загрузки, очистки, кэширования и подготовки батчей текста.
Поддерживает динамическую токенизацию и версионирование кэша на основе
хэширования конфигурации очистки.
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

    Отвечает за полный жизненный цикл данных: от загрузки сырых файлов (HF/CSV)
    до выдачи готовых тензоров через DataLoader. Реализует паттерн умного
    кэширования: результаты очистки сохраняются на диск с уникальным хэшем,
    зависящим от конфигурации клинеров.

    Attributes:
        data_cfg (Any): Секция конфигурации данных (DictConfig или Pydantic схема).
        tokenizer (PreTrainedTokenizerBase): Инициализированный токенизатор HuggingFace.
        processed_dir (str): Путь к директории с закэшированными очищенными данными.
    """

    def __init__(self, data_cfg: Any, tokenizer: PreTrainedTokenizerBase):
        """
        Инициализирует DataModule и вычисляет путь для DVC кэша.

        Args:
            data_cfg: Настройки данных (источник, клинеры, параметры DataLoader'а).
            tokenizer: Объект токенизатора. Передается снаружи для избежания
                циклических зависимостей между конфигами модели и данных.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        
        # Вычисляем MD5 хэш от параметров очистки для инвалидации кэша.
        # Если конфиг `cleaner` изменится, хэш поменяется, и скрипт создаст новую папку.
        cleaner_dict = OmegaConf.to_container(self.data_cfg.cleaner, resolve=True)
        cleaner_str = json.dumps(cleaner_dict, sort_keys=True)
        config_hash = hashlib.md5(cleaner_str.encode('utf-8')).hexdigest()[:8]
        
        self.processed_dir = os.path.join(
            self.data_cfg.paths.processed_data_dir, 
            f"{self.data_cfg.dataset_name}_cleaned_{config_hash}"
        )

    def prepare_data(self) -> None:
        """
        Скачивает сырые данные, применяет очистку и сохраняет результат на диск.

        Примечание:
            В PyTorch Lightning этот метод выполняется СТРОГО на одном процессе (GPU 0).
            Здесь нельзя присваивать атрибуты класса (self.train_dataset = ...),
            так как другие процессы (GPU 1, 2...) их не увидят. Только работа с диском.
        """
        # Проверяем наличие кэша. Флаг force_reprocess позволяет принудительно переписать кэш.
        if os.path.exists(self.processed_dir) and not self.data_cfg.get("force_reprocess", False):
            logger.info(f"Нашли кэш данных: {self.processed_dir}. Очистка пропущена.")
            return

        logger.info("Начинаем загрузку и обработку сырых данных...")
        
        # Загрузка через HF datasets (поддерживает как локальные файлы, так и HF Hub)
        raw_datasets = instantiate(self.data_cfg.source)
        
        # Обработка бизнес-данных без явных сплитов (например, цельного CSV)
        if "validation" in raw_datasets:
            raw_train, raw_val = raw_datasets["train"], raw_datasets["validation"]
        else:
            split_ds = raw_datasets["train"].train_test_split(
                test_size=self.data_cfg.val_split_size, 
                seed=self.data_cfg.seed
            )
            raw_train, raw_val = split_ds["train"], split_ds["test"]

        # Инициализация цепочки фильтров и адаптеров
        cleaner_pipeline = instantiate(self.data_cfg.cleaner)

        clean_train = NLPDatasetAdapter(
            hf_dataset=raw_train, 
            text_column=self.data_cfg.text_column, 
            cleaning_pipeline=cleaner_pipeline,
            num_proc=self.data_cfg.get("preprocessing_num_workers", 4),
            batch_size=self.data_cfg.get("preprocessing_batch_size", 1000)
        ).prepare_dataset()

        clean_val = NLPDatasetAdapter(
            hf_dataset=raw_val, 
            text_column=self.data_cfg.text_column, 
            cleaning_pipeline=cleaner_pipeline,
            num_proc=self.data_cfg.get("preprocessing_num_workers", 4),
            batch_size=self.data_cfg.get("preprocessing_batch_size", 1000)
        ).prepare_dataset()

        # Сохранение в оптимизированном формате Apache Arrow
        processed_dataset = DatasetDict({
            "train": clean_train,
            "validation": clean_val
        })
        processed_dataset.save_to_disk(self.processed_dir)
        logger.info(f"Данные успешно очищены и сохранены в {self.processed_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Загружает обработанные данные с диска в память текущего процесса.

        Примечание:
            Выполняется на КАЖДОМ процессе (на каждой видеокарте).

        Args:
            stage: Текущая стадия PyTorch Lightning ('fit', 'validate', 'test', 'predict').
        """
        processed_dataset = load_from_disk(self.processed_dir)
        
        if stage == "fit" or stage is None:
            self.train_dataset = processed_dataset["train"]
            self.val_dataset = processed_dataset["validation"]

        # Инициализируем коллатор динамически через Гидру, прокидывая токенизатор в kwargs
        self.collator = instantiate(
            self.data_cfg.collator, 
            tokenizer=self.tokenizer
        )

    def train_dataloader(self) -> Any:
        """
        Создает DataLoader для тренировочной выборки.
        Настройки (batch_size, num_workers) берутся из конфигурации Гидры.
        """
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.train_dataset,
            collate_fn=self.collator,
            shuffle=True
        )

    def val_dataloader(self) -> Any:
        """
        Создает DataLoader для валидационной выборки.
        """
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.val_dataset,
            collate_fn=self.collator,
            shuffle=False
        )