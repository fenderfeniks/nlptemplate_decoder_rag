# src/pipelines/base/core/data/transforms/validation.py
import logging
from typing import Any, Optional

from datasets import Dataset as HFDataset
from pydantic import ValidationError
from src.pipelines.base.core.data.transforms.base import BaseDatasetTransform
from src.pipelines.base.core.data.cleaners import TextCleaningPipeline
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

class BaseValidationTransform(BaseDatasetTransform, ABC):
    """Базовый класс для Pydantic-валидации датасетов."""

    def __init__(self, mode: str, num_proc: int = 4, batch_size: int = 1000) -> None:
        self.mode = mode
        self.num_proc = num_proc
        self.batch_size = batch_size
        self._validate_mode()

    @abstractmethod
    def _validate_mode(self) -> None:
        """Проверка валидности переданного mode при инициализации."""
        pass

    @abstractmethod
    def _get_required_columns(self) -> list[str]:
        """Возвращает список колонок, необходимых для текущего mode."""
        pass

    @abstractmethod
    def _get_filter_column(self) -> str:
        """Возвращает название колонки, по которой будет идти финальный filter."""
        pass

    @abstractmethod
    def _validate_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Применяет Pydantic-схему к батчу."""
        pass

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Применение Pydantic-валидации (режим: %s)...", self.mode)
        initial_count = len(dataset)

        required_columns = self._get_required_columns()
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        
        if missing_columns:
            logger.warning(
                "Колонки %s не найдены в датасете — валидация пропущена. "
                "Убедитесь, что режим '%s' соответствует составу датасета.",
                missing_columns,
                self.mode,
            )
            return dataset

        dataset = dataset.map(
            self._validate_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc=f"Validating {self.mode} records",
        )
        
        filter_col = self._get_filter_column()
        dataset = dataset.filter(
            lambda x: bool(x[filter_col]),
            num_proc=self.num_proc,
        )

        logger.info(
            "Валидация завершена: %d → %d записей (отброшено %d)",
            initial_count,
            len(dataset),
            initial_count - len(dataset),
        )
        return dataset
    

class CleaningTransform(BaseDatasetTransform):
    """Трансформация для очистки текста через кастомные клинеры.

    Список ``columns_to_clean`` может содержать ``None`` (Hydra интерполирует
    отсутствующие поля как ``null``). Такие значения фильтруются автоматически.
    Колонки, которых нет в датасете, пропускаются без ошибки.
    """

    def __init__(
        self,
        pipeline: TextCleaningPipeline,
        columns_to_clean: list[Optional[str]],
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        """
        Args:
            pipeline: Инстанс ``TextCleaningPipeline`` с набором клинеров.
            columns_to_clean: Список имён колонок для очистки. Значения ``None``
                (например, от Hydra-интерполяции null-полей) игнорируются.
            num_proc: Число процессов для параллельного map.
            batch_size: Размер батча для map.
        """
        self.pipeline = pipeline
        # Убираем None сразу при инициализации — нет смысла проверять их каждый раз
        self.columns_to_clean: list[str] = [c for c in columns_to_clean if c is not None]
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        active_cols = [c for c in self.columns_to_clean if c in dataset.column_names]

        if not active_cols:
            logger.warning(
                "Ни одна из колонок %s не найдена в датасете — очистка пропущена.",
                self.columns_to_clean,
            )
            return dataset

        logger.info("Применение пайплайна очистки текста по колонкам: %s...", active_cols)

        def _clean_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            return {
                col: [self.pipeline(t) for t in batch[col]]
                for col in active_cols
            }

        result = dataset.map(
            _clean_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Cleaning text",
        )

        logger.info("Очистка текста завершена: обработано %d записей.", len(result))
        return result