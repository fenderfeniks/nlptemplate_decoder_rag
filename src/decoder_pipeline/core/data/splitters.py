# src/decoder_pipeline/core/data/splitters.py
import logging

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class RandomDatasetSplitter:
    """Разбивает датасет на train, validation и test сплиты.

    Если датасет уже содержит сплиты ``validation`` и ``test`` (например,
    скачан с HuggingFace Hub), разбиение пропускается.

    Логика двухшагового разбиения:
    1. Отделяем holdout (val + test) от train.
    2. Делим holdout на val и test в нужной пропорции.

    Это гарантирует что train, val и test не пересекаются даже при разных
    значениях ``val_size`` и ``test_size``.
    """

    def __init__(
        self,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            val_size: Доля данных для валидационного сплита [0, 1).
            test_size: Доля данных для тестового сплита [0, 1).
            seed: Seed для воспроизводимого разбиения.

        Raises:
            ValueError: Если val_size или test_size отрицательны,
                или их сумма >= 1.0 (не остаётся данных на train).
        """
        if val_size < 0 or test_size < 0:
            raise ValueError(
                f"val_size и test_size должны быть >= 0. "
                f"Получено: val_size={val_size}, test_size={test_size}."
            )
        if val_size + test_size >= 1.0:
            raise ValueError(
                f"val_size + test_size должны быть < 1.0 (иначе train пуст). "
                f"Получено: {val_size + test_size:.3f}."
            )
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def __call__(self, raw_datasets: Dataset | DatasetDict) -> DatasetDict:
        """Применяет разбиение к датасету.

        Args:
            raw_datasets: Исходный Dataset или DatasetDict.

        Returns:
            DatasetDict с ключами 'train' и опционально 'validation', 'test'.
        """
        # Если оба сплита уже есть — пропускаем разбиение целиком
        if (
            isinstance(raw_datasets, DatasetDict)
            and "validation" in raw_datasets
            and "test" in raw_datasets
        ):
            logger.info(
                "Датасет уже содержит сплиты train/validation/test. Разбиение пропущено."
            )
            return raw_datasets

        base_ds = (
            raw_datasets["train"]
            if isinstance(raw_datasets, DatasetDict)
            else raw_datasets
        )

        total_holdout = self.val_size + self.test_size

        # Оба нуля — возвращаем только train (например, для indexing-задачи)
        if total_holdout == 0.0:
            logger.info(
                "val_size=0 и test_size=0: возвращаем только train (%d записей).",
                len(base_ds),
            )
            return DatasetDict({"train": base_ds})

        logger.info(
            "Разбиение датасета: val_size=%.3f, test_size=%.3f, seed=%d, всего=%d",
            self.val_size, self.test_size, self.seed, len(base_ds),
        )

        # Шаг 1: train vs (val + test)
        split_1 = base_ds.train_test_split(
            test_size=total_holdout,
            seed=self.seed,
            shuffle=True,
        )

        result = DatasetDict({"train": split_1["train"]})

        if self.val_size == 0.0:
            # Только test, val не нужен
            result["test"] = split_1["test"]
            logger.info(
                "train=%d, test=%d (val пропущен)",
                len(result["train"]), len(result["test"]),
            )
            return result

        if self.test_size == 0.0:
            # Только val, test не нужен
            result["validation"] = split_1["test"]
            logger.info(
                "train=%d, validation=%d (test пропущен)",
                len(result["train"]), len(result["validation"]),
            )
            return result

        # Шаг 2: val vs test из общего holdout
        test_proportion = self.test_size / total_holdout
        split_2 = split_1["test"].train_test_split(
            test_size=test_proportion,
            seed=self.seed,
            shuffle=True,
        )

        result["validation"] = split_2["train"]
        result["test"] = split_2["test"]

        logger.info(
            "Разбиение завершено: train=%d, validation=%d, test=%d",
            len(result["train"]),
            len(result["validation"]),
            len(result["test"]),
        )
        return result