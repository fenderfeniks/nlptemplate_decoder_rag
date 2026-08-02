# src/decoder_pipeline/core/data/transforms/filtering.py
import logging

from datasets import Dataset as HFDataset

from src.decoder_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class LengthFilterTransform(BaseDatasetTransform):
    """Трансформация для отсечения слишком длинных токенизированных последовательностей.

    Ожидает, что в датасете уже присутствует колонка ``input_ids`` (или другая,
    заданная через ``column``). Применяется после токенизации.
    """

    def __init__(
        self,
        max_length: int = 2048,
        column: str = "input_ids",
        num_proc: int = 4,
    ) -> None:
        """
        Args:
            max_length: Максимально допустимая длина последовательности в токенах.
                Должен быть положительным числом.
            column: Колонка, по длине которой производится фильтрация.
                По умолчанию ``input_ids``.
            num_proc: Число процессов для параллельной фильтрации.

        Raises:
            ValueError: Если ``max_length`` не является положительным числом.
        """
        if max_length <= 0:
            raise ValueError(
                f"max_length должен быть положительным числом, получено: {max_length}"
            )
        self.max_length = max_length
        self.column = column
        self.num_proc = num_proc

    def __call__(self, dataset: HFDataset) -> HFDataset:
        if self.column not in dataset.column_names:
            logger.warning(
                "Колонка '%s' не найдена в датасете — фильтрация по длине пропущена. "
                "Убедитесь, что токенизация выполнена до этого шага.",
                self.column,
            )
            return dataset

        initial_count = len(dataset)
        filtered_ds = dataset.filter(
            lambda x: len(x[self.column]) <= self.max_length,
            num_proc=self.num_proc,
            desc=f"Filtering > {self.max_length} tokens by '{self.column}'",
        )
        removed = initial_count - len(filtered_ds)
        logger.info(
            "Фильтрация по длине ('%s' <= %d): %d -> %d (удалено %d)",
            self.column,
            self.max_length,
            initial_count,
            len(filtered_ds),
            removed,
        )
        return filtered_ds