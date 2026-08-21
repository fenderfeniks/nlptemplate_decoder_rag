# src/pipelines/base/core/data/splitters.py
import logging

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class RandomDatasetSplitter:
    """Разбивает датасет на train и validation сплиты.

    Сплит test исключен из динамического формирования. Эталонные данные 
    (test) должны загружаться извне (из фиксированного бенчмарка) для 
    обеспечения строгой воспроизводимости и предотвращения утечек.
    """

    def __init__(
        self,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            val_size: Доля данных для валидационного сплита [0, 1).
            seed: Seed для воспроизводимого разбиения.

        Raises:
            ValueError: Если val_size отрицательный или >= 1.0.
        """
        if val_size < 0 or val_size >= 1.0:
            raise ValueError(
                f"val_size должен быть в диапазоне [0, 1). "
                f"Получено: val_size={val_size}."
            )
        self.val_size = val_size
        self.seed = seed

    def __call__(self, raw_datasets: Dataset | DatasetDict) -> DatasetDict:
        """Применяет разбиение к датасету.

        Args:
            raw_datasets: Исходный Dataset или DatasetDict.

        Returns:
            DatasetDict с ключами 'train' и опционально 'validation'.
        """
        if (
            isinstance(raw_datasets, DatasetDict)
            and "validation" in raw_datasets
        ):
            logger.info(
                "Датасет уже содержит сплит validation. Разбиение пропущено."
            )
            # Оставляем только train и val, игнорируя встроенный test если он был
            return DatasetDict({
                "train": raw_datasets["train"],
                "validation": raw_datasets["validation"]
            })

        base_ds = (
            raw_datasets["train"]
            if isinstance(raw_datasets, DatasetDict)
            else raw_datasets
        )

        if self.val_size == 0.0:
            logger.info(
                "val_size=0: возвращаем только train (%d записей).",
                len(base_ds),
            )
            return DatasetDict({"train": base_ds})

        logger.info(
            "Разбиение датасета: val_size=%.3f, seed=%d, всего=%d",
            self.val_size, self.seed, len(base_ds),
        )

        splits = base_ds.train_test_split(
            test_size=self.val_size,
            seed=self.seed,
            shuffle=True,
        )

        result = DatasetDict({
            "train": splits["train"],
            "validation": splits["test"]
        })

        logger.info(
            "Разбиение завершено: train=%d, validation=%d",
            len(result["train"]),
            len(result["validation"]),
        )
        return result