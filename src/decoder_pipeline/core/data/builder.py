# src/decoder_pipeline/core/data/builder.py
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
from datasets import DatasetDict, load_from_disk
from hydra.utils import instantiate
from omegaconf import OmegaConf, ListConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Ключ, по которому в _target_ определяем трансформы, требующие tokenizer
_TOKENIZATION_MARKER = "TokenizationTransform"

class NLPDataModule(pl.LightningDataModule):
    """Универсальный DataModule для работы с RAG-датасетами.

    Поддерживает режимы 'indexing' и 'contrastive' за счет делегирования 
    работы пайплайну трансформаций. Обработанные данные кэшируются на диске.
    """

    def __init__(self, data_cfg: Any, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        self.processed_dir = self._resolve_processed_dir()

    # ------------------------------------------------------------------
    # Кэш-хэш
    # ------------------------------------------------------------------

    def _resolve_processed_dir(self) -> Path:
        """Вычисляет путь к кэшу обработанных данных по SHA-256 хэшу конфига."""
        hash_dict = {
            "source": OmegaConf.to_container(self.data_cfg.source, resolve=True),
            "splitter": OmegaConf.to_container(self.data_cfg.splitter, resolve=True),
            # transforms — явный list[DictConfig], порядок гарантирован
            "transforms": OmegaConf.to_container(self.data_cfg.transforms, resolve=True),
            "seed": self.data_cfg.get("seed"),
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", "custom_tokenizer"),
        }
        # SHA-256 надёжнее MD5 и не имеет deprecation-предупреждений в новых Python
        hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:8]

        dataset_name = self.data_cfg.get("dataset_name", "nlp_dataset")
        return (
            Path(self.data_cfg.paths.processed_data_dir)
            / f"{dataset_name}_processed_{config_hash}"
        )
    
    # ------------------------------------------------------------------
    # Подвыборка
    # ------------------------------------------------------------------

    def _maybe_subsample(self, dataset: Any, name: str) -> Any:
        max_samples = self.data_cfg.get("max_samples", None)
        if max_samples is None:
            return dataset

        n = len(dataset)
        k = max(1, int(n * max_samples)) if isinstance(max_samples, float) else min(int(max_samples), n)

        if k >= n:
            return dataset
        
        seed = self.data_cfg.get("seed", 42)

        logger.info(
                    "max_samples (%s): %s %d → %d записей (seed=%d)",
                    max_samples, name, n, k, seed,
                )
                # Перемешиваем перед выборкой — иначе берём только первые k строк,
                # что смещает выборку если датасет отсортирован (по дате, классу и т.п.)
        shuffled = dataset.shuffle(seed=seed)
        return shuffled.select(range(k))

    # ------------------------------------------------------------------
    # Трансформации
    # ------------------------------------------------------------------

    def _build_transforms(self) -> list[Any]:
        """Инстанцирует трансформации в порядке, заданном конфигом.

        Конфиг ``transforms`` должен быть **списком** (``ListConfig``), а не словарём.
        Это единственный способ гарантировать порядок применения независимо от того,
        как собирался конфиг — через Hydra defaults или программно.

        Трансформы с ``_target_``, содержащим ``TokenizationTransform``,
        получают ``tokenizer`` как дополнительный аргумент.

        Returns:
            Список инстанцированных трансформаций.

        Raises:
            TypeError: Если ``data_cfg.transforms`` — не список.
        """
        transforms_cfg = self.data_cfg.transforms

        if OmegaConf.is_dict(transforms_cfg):
            transforms_cfg = list(transforms_cfg.values())

        if not isinstance(transforms_cfg, (list, ListConfig)):
            raise TypeError(
                "data_cfg.transforms должен быть списком (transforms: [...]), "
                f"получено: {type(transforms_cfg).__name__}. "
                "Измените конфиг: вместо именованных ключей используйте явный list."
            )

        transforms = []
        for i, transform_cfg in enumerate(transforms_cfg):
            target = transform_cfg.get("_target_", "")
            if _TOKENIZATION_MARKER in target:
                instance = instantiate(transform_cfg, tokenizer=self.tokenizer)
            else:
                instance = instantiate(transform_cfg)

            logger.debug("Трансформ [%d]: %s", i, target)
            transforms.append(instance)

        logger.info(
            "Инициализировано %d трансформаций: %s",
            len(transforms),
            [t.get("_target_", "?") for t in transforms_cfg],
        )
        return transforms

    def _apply_transforms(self, dataset_split: Any, transforms: list[Any]) -> Any:
        for transform in transforms:
            dataset_split = transform(dataset_split)
        return dataset_split
        
    # ------------------------------------------------------------------
    # LightningDataModule hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        if self.processed_dir.exists() and not self.data_cfg.get("force_reprocess", False):
            logger.info("Нашли кэш обработанных данных: %s. Подготовка пропущена.", self.processed_dir)
            return

        logger.info("Начинаем загрузку и применение трансформаций RAG...")

        fetcher = instantiate(self.data_cfg.source)
        raw_datasets = fetcher.load()

        splitter = instantiate(self.data_cfg.splitter)
        split_datasets = splitter(raw_datasets)

        transforms = self._build_transforms()

        available_splits = list(split_datasets.keys())
        logger.info("Доступные сплиты для обработки: %s", available_splits)

        # 5. Применяем трансформации и подвыборку к каждому сплиту
        processed: dict[str, Any] = {}
        for split_name in available_splits:
            logger.info("Обработка сплита '%s'...", split_name)
            subsampled = self._maybe_subsample(split_datasets[split_name], split_name)
            processed[split_name] = self._apply_transforms(subsampled, transforms)

        processed_dataset = DatasetDict(processed)
        processed_dataset.save_to_disk(str(self.processed_dir))
        logger.info(
            "Данные обработаны и сохранены в %s. Размеры: %s",
            self.processed_dir,
            {k: len(v) for k, v in processed_dataset.items()},
        )

    def setup(self, stage: Optional[str] = None) -> None:
        processed_dataset = load_from_disk(str(self.processed_dir))

        if stage in ("fit", None):
            self.train_dataset = processed_dataset["train"]
            self.val_dataset = processed_dataset.get("validation")
        if stage in ("validate", None):
            self.val_dataset = processed_dataset.get("validation")
        if stage in ("test", None):
            self.test_dataset = processed_dataset.get("test")

        self.collator = instantiate(self.data_cfg.collator, tokenizer=self.tokenizer)

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------
   
    def _dataloader_kwargs(self) -> dict:
        dl_cfg = OmegaConf.to_container(self.data_cfg.dataloader, resolve=True)
        for key in ("_target_", "dataset", "collate_fn", "shuffle"):
            dl_cfg.pop(key, None)
        return dl_cfg

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collator,
            shuffle=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            logger.debug("val_dataset не задан — val_dataloader пропущен.")
            return None
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collator,
            shuffle=False,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            logger.debug("test_dataset не задан — test_dataloader пропущен.")
            return None
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collator,
            shuffle=False,
            **self._dataloader_kwargs(),
        )