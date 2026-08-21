# src/pipelines/base/core/data/builder.py
"""Универсальный DataModule для NLP-задач (SFT, CPT, RAG indexing/contrastive).

Архитектура сплитов:
    train / val  — данные из raw источника. Проходят полный пайплайн трансформаций
                   (cleaning, dedup, BenchmarkExclusionTransform, tokenization,
                   length filter). Val используется для мониторинга loss во время
                   обучения — поэтому обрабатывается идентично train.

    test         — берётся из зафиксированного бенчмарка (benchmark_uri в манифесте).
                   Хранится как test_dataset_raw (сырые тексты, без токенизации).
                   Используется только для генерации и подсчёта бизнес-метрик
                   (ROUGE, BLEU и т.д.). Loss на тесте не считается.

Сырые датасеты для генерации в GenerationEvaluationCallback:
    val_dataset_raw  — сырой val до трансформаций, колонки из data_cfg
                       (prompt_column / target_column). Сохраняется на диск
                       рядом с токенизированным кэшем.
    test_dataset_raw — сырой бенчмарк, колонки из data_cfg.eval
                       (prompt_column / target_column). Только в памяти.

Конфиг eval:
    data_cfg.eval — подключается через defaults в train_decoder.yaml:
                    - eval: eval_decoder
                    Определяет prompt_column/target_column для бенчмарка.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional
import shutil

import pytorch_lightning as pl
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_TOKENIZATION_MARKER = "TokenizationTransform"
_EXCLUSION_MARKER = "BenchmarkExclusionTransform"

_RAW_VAL_SUFFIX = "_val_raw"


class DataModule(pl.LightningDataModule):
    """Универсальный DataModule для NLP-задач.

    Args:
        data_cfg:         Hydra DictConfig секции data (включает data.eval).
        processed_data_dir: Директория для кэша обработанных данных.
        tokenizer:        Токенизатор — пробрасывается в TokenizationTransform.
        benchmark_loader: Опциональный BenchmarkLoader для загрузки test_dataset_raw
                          из манифеста. Если None — тест недоступен.
    """

    def __init__(
        self,
        data_cfg: Any,
        processed_data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        benchmark_loader: Any | None = None,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        self.benchmark_loader = benchmark_loader
        self.processed_data_dir = processed_data_dir

        # Токенизированные датасеты — для DataLoader'ов (train/val)
        self.train_dataset = None
        self.val_dataset = None

        # Сырые датасеты — для генерации в GenerationEvaluationCallback
        # val_dataset_raw: сохраняется на диск, колонки data_cfg.prompt_column/target_column
        # test_dataset_raw: только в памяти, колонки из data_cfg.eval
        self.val_dataset_raw: HFDataset | None = None
        self.test_dataset_raw: HFDataset | None = None
        self.test_dataset: HFDataset | None = None

        self.collator = None
        self.processed_dir = self._resolve_processed_dir()
        self.raw_val_dir = (
            self.processed_dir.parent / f"{self.processed_dir.name}{_RAW_VAL_SUFFIX}"
        )

    # ------------------------------------------------------------------
    # Кэш-хэш
    # ------------------------------------------------------------------

    def _resolve_processed_dir(self) -> Path:
        """Путь к кэшу обработанных данных по SHA-256 хэшу конфига."""
        hash_dict = {
            "source": OmegaConf.to_container(self.data_cfg.source, resolve=True) if "source" in self.data_cfg else {},
            "splitter": OmegaConf.to_container(self.data_cfg.splitter, resolve=True) if "splitter" in self.data_cfg else {},
            "transforms": OmegaConf.to_container(self.data_cfg.transforms, resolve=True) if "transforms" in self.data_cfg else {},
            "seed": self.data_cfg.get("seed"),
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", "custom_tokenizer"),
        }
        hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:8]
        dataset_name = self.data_cfg.get("dataset_name", "nlp_dataset")
        return Path(self.processed_data_dir) / f"{dataset_name}_processed_{config_hash}"

    # ------------------------------------------------------------------
    # Трансформации
    # ------------------------------------------------------------------

    def _build_transforms(self, transforms_cfg: Any) -> list[Any]:
        if OmegaConf.is_dict(transforms_cfg):
            transforms_list = [transforms_cfg[k] for k in transforms_cfg]
        elif isinstance(transforms_cfg, (list, ListConfig)):
            transforms_list = list(transforms_cfg)
        else:
            raise TypeError(
                "transforms должен быть словарём или списком, "
                f"получено: {type(transforms_cfg).__name__}."
            )

        transforms = []
        for i, transform_cfg in enumerate(transforms_list):
            target = transform_cfg.get("_target_", "")
            if _TOKENIZATION_MARKER in target:
                instance = instantiate(transform_cfg, tokenizer=self.tokenizer)
            elif _EXCLUSION_MARKER in target:
                instance = instantiate(
                    transform_cfg,
                    benchmark_loader=self.benchmark_loader,
                )
            else:
                instance = instantiate(transform_cfg)
            logger.debug("Трансформ [%d]: %s", i, target)
            transforms.append(instance)

        logger.info(
            "Инициализировано %d трансформаций: %s",
            len(transforms),
            [transforms_list[i].get("_target_", "?") for i in range(len(transforms_list))],
        )
        return transforms

    def _apply_transforms(self, dataset_split: Any, transforms: list[Any]) -> Any:
        for transform in transforms:
            dataset_split = transform(dataset_split)
        return dataset_split

    # ------------------------------------------------------------------
    # Подвыборка
    # ------------------------------------------------------------------

    def _maybe_subsample(self, dataset: Any, split_name: str) -> Any:
        max_samples = self.data_cfg.get("max_samples", None)
        if max_samples is None:
            return dataset

        n = len(dataset)
        k = (
            max(1, int(n * max_samples))
            if isinstance(max_samples, float)
            else min(int(max_samples), n)
        )
        if k >= n:
            return dataset

        seed = self.data_cfg.get("seed", 42)
        logger.info(
            "max_samples (%s): %s %d -> %d записей (seed=%d)",
            max_samples, split_name, n, k, seed,
        )
        return dataset.shuffle(seed=seed).select(range(k))

    # ------------------------------------------------------------------
    # prepare_data: train + val из raw
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        if getattr(self, "_prepare_data_done", False):
            logger.info("prepare_data уже выполнена в этой сессии — пропускаем.")
            return

        force = self.data_cfg.get("force_reprocess", False)

        if self.processed_dir.exists() and not force:
            logger.info("Кэш найден: %s — prepare_data пропущена.", self.processed_dir)
            self._prepare_data_done = True
            return

        logger.info("Запуск пайплайна подготовки данных (train/val)...")

        fetcher = instantiate(self.data_cfg.source)
        raw_datasets = fetcher.load()

        splitter_cfg = self.data_cfg.get("splitter")
        if splitter_cfg:
            splitter = instantiate(splitter_cfg)
            split_datasets = splitter(raw_datasets)
        else:
            logger.info("Сплиттер не задан. Весь датасет направлен в 'train'.")
            if isinstance(raw_datasets, (dict, DatasetDict)):
                split_datasets = raw_datasets
            else:
                split_datasets = {"train": raw_datasets}

        transforms = self._build_transforms(self.data_cfg.get("transforms", {}))
        
        if self.benchmark_loader is not None:
            local_benchmark_path = self.benchmark_loader.resolve_local_path()
            if local_benchmark_path is not None:
                for t in transforms:
                    if hasattr(t, "benchmark_path"):
                        t.benchmark_path = local_benchmark_path
                        logger.info(
                            "BenchmarkExclusionTransform: путь обновлён → %s",
                            local_benchmark_path,
                        )

        train_val_splits = {
            k: v for k, v in split_datasets.items() if k in ("train", "validation")
        }
        logger.info("Обрабатываемые сплиты: %s", list(train_val_splits.keys()))

        # Сохраняем сырой val ДО трансформаций — только колонки для генерации
        raw_val = train_val_splits.get("validation")
        if raw_val is not None:
            # Собираем все возможные колонки (и для SFT, и для RAG)
            candidates = [
                self.data_cfg.get("prompt_column"),
                self.data_cfg.get("target_column"),
                self.data_cfg.get("query_column"),
                self.data_cfg.get("positive_column"),
                self.data_cfg.get("negative_column"),
            ]
            
            # Убираем None и оставляем только те, что реально есть в датасете
            cols_to_keep = list({c for c in candidates if c and c in raw_val.column_names})

            if not cols_to_keep:
                logger.warning(
                    "Ни одна из целевых колонок не найдена. Сохраняем raw_val целиком."
                )
                raw_val_slim = raw_val
            else:
                raw_val_slim = raw_val.select_columns(cols_to_keep)

            if self.raw_val_dir.exists():
                shutil.rmtree(self.raw_val_dir)
            raw_val_slim.save_to_disk(str(self.raw_val_dir))
            logger.info(
                "Сырой val сохранён: %s (%d записей, колонки: %s)",
                self.raw_val_dir, len(raw_val_slim), cols_to_keep if cols_to_keep else "все",
            )

        processed: dict[str, Any] = {}
        for split_name, split_data in train_val_splits.items():
            logger.info("Обработка сплита '%s'...", split_name)
            subsampled = self._maybe_subsample(split_data, split_name)
            processed[split_name] = self._apply_transforms(subsampled, transforms)

        if self.processed_dir.exists():
            logger.info("Удаляем старый кэш: %s", self.processed_dir)
            shutil.rmtree(self.processed_dir)

        DatasetDict(processed).save_to_disk(str(self.processed_dir))
        self._prepare_data_done = True
        logger.info(
            "Train/val сохранены в %s. Размеры: %s",
            self.processed_dir,
            {k: len(v) for k, v in processed.items()},
        )

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Загружает датасеты для нужной стадии.

        fit:  train_dataset, val_dataset (токенизированные из кэша)
              val_dataset_raw (сырой val для генерации)
        test: test_dataset_raw (сырой бенчмарк для генерации)
        """
        self.collator = instantiate(self.data_cfg.collator, tokenizer=self.tokenizer)

        if stage in ("fit", None):
            processed = load_from_disk(str(self.processed_dir))
            self.train_dataset = processed["train"]
            self.val_dataset = processed.get("validation")
            logger.info(
                "setup('fit'): train=%d, val=%s",
                len(self.train_dataset),
                len(self.val_dataset) if self.val_dataset else "нет",
            )

            # Сырой val для генерации в колбэке
            if self.raw_val_dir.exists():
                self.val_dataset_raw = load_from_disk(str(self.raw_val_dir))
                logger.info(
                    "val_dataset_raw: %d записей из %s",
                    len(self.val_dataset_raw), self.raw_val_dir,
                )
            else:
                logger.warning(
                    "Сырой val не найден (%s) — генерация на val недоступна. "
                    "Запустите prepare_data заново.",
                    self.raw_val_dir,
                )

        if stage in ("validate", None) and self.val_dataset is None:
            processed = load_from_disk(str(self.processed_dir))
            self.val_dataset = processed.get("validation")

        if stage in ("test", None):
            self._setup_test()

    def _setup_test(self) -> None:
        """Загружает и опционально токенизирует бенчмарк для теста."""
        if self.benchmark_loader is None:
            logger.info("BenchmarkLoader не передан — тест недоступен.")
            return

        # 1. Получаем имена колонок из бенчмарка (из eval-секции или дефолтные)
        eval_cfg = self.data_cfg.get("eval", {})
        benchmark_query = eval_cfg.get("prompt_column", "prompt")
        benchmark_answer = eval_cfg.get("target_column", "response")

        benchmark_ds = self.benchmark_loader.load_as_dataset(
            query_column=benchmark_query,
            answer_column=benchmark_answer,
            doc_id_column=self.data_cfg.get("id_column", "chunk_id"),
        )

        if benchmark_ds is None:
            logger.warning("Бенчмарк не найден — тест недоступен.")
            return

        missing = [c for c in [benchmark_query, benchmark_answer] if c not in benchmark_ds.column_names]
        if missing:
            logger.error(
                "Бенчмарк не содержит колонки %s (доступны: %s).",
                missing, benchmark_ds.column_names,
            )
            return

        self.test_dataset_raw = benchmark_ds
        logger.info(
            "test_dataset_raw: %d записей, колонки: %s",
            len(self.test_dataset_raw), self.test_dataset_raw.column_names,
        )

        # 2. Адаптируем имена колонок под ожидаемые в трансформах (для test_dataloader)
        logger.info("Подготовка токенизированного test_dataset...")
        train_query = self.data_cfg.get("prompt_column") or self.data_cfg.get("query_column")
        train_answer = self.data_cfg.get("target_column") or self.data_cfg.get("positive_column")
        
        ds_for_transforms = benchmark_ds
        rename_map = {}
        if train_query and train_query != benchmark_query:
            rename_map[benchmark_query] = train_query
        if train_answer and train_answer != benchmark_answer:
            rename_map[benchmark_answer] = train_answer
            
        if rename_map:
            ds_for_transforms = ds_for_transforms.rename_columns(rename_map)
            logger.debug("Колонки бенчмарка переименованы для трансформаций: %s", rename_map)

        # 3. Применяем трансформации
        transforms = self._build_transforms(self.data_cfg.transforms)
        transforms = [t for t in transforms if _EXCLUSION_MARKER not in t.__class__.__name__]
        self.test_dataset = self._apply_transforms(ds_for_transforms, transforms)
        
    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _dataloader_kwargs(self) -> dict:
        dl_cfg = OmegaConf.to_container(self.data_cfg.dataloader, resolve=True)
        for key in ("_target_", "dataset", "collate_fn", "shuffle"):
            dl_cfg.pop(key, None)
        return dl_cfg

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset не инициализирован. Вызовите setup('fit').")
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
        if getattr(self, "test_dataset", None) is None:
            logger.debug("test_dataset не токенизирован — test_dataloader пропущен.")
            return None
        
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collator,
            shuffle=False,
            **self._dataloader_kwargs(),
        )