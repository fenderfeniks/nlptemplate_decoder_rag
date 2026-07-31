# src/core/data/mixers.py
import logging
from typing import Any

from datasets import Dataset, DatasetDict, interleave_datasets

logger = logging.getLogger(__name__)

_VALID_STOPPING_STRATEGIES = ("first_exhausted", "all_exhausted")


class InterleavedDataFetcher:
    """Смешивает несколько источников данных с заданными вероятностями.

    Является прозрачной заменой ``RawDataFetcher`` — имеет тот же публичный
    метод ``load()`` и возвращает ``DatasetDict`` с ключом ``'train'``,
    который затем поступает в ``RandomDatasetSplitter``.

    .. note::
        При ``stopping_strategy='first_exhausted'`` объём смешанного датасета
        ограничен наименьшим источником умноженным на его вероятность. Если датасеты
        сильно различаются по размеру — логируем предупреждение.
    """

    def __init__(
        self,
        fetchers: list[Any],
        probabilities: list[float],
        seed: int = 42,
        stopping_strategy: str = "first_exhausted",
        imbalance_warning_ratio: float = 10.0,
    ) -> None:
        """
        Args:
            fetchers: Список инстанцированных загрузчиков (RawDataFetcher или другие
                объекты с методом ``load() -> Dataset | DatasetDict``).
            probabilities: Вероятности выборки из каждого датасета.
                Должны быть > 0 и суммироваться в 1.0 (±1e-6).
            seed: Seed для воспроизводимого смешивания.
            stopping_strategy: Стратегия остановки:
                - ``'first_exhausted'``: стоп, когда закончится наименьший датасет;
                - ``'all_exhausted'``: стоп, когда закончатся все (с повторениями).

        Raises:
            ValueError: При несовпадении длин, невалидных вероятностях
                или неизвестной stopping_strategy.
        """
        if len(fetchers) != len(probabilities):
            raise ValueError(
                f"Число fetchers ({len(fetchers)}) должно совпадать "
                f"с числом probabilities ({len(probabilities)})."
            )
        if not fetchers:
            raise ValueError("fetchers не может быть пустым списком.")
        if any(p <= 0 for p in probabilities):
            raise ValueError(
                f"Все вероятности должны быть > 0. Получено: {probabilities}."
            )
        prob_sum = sum(probabilities)
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Сумма probabilities должна быть равна 1.0. "
                f"Получено: {prob_sum:.8f} (разница {abs(prob_sum - 1.0):.2e})."
            )
        if stopping_strategy not in _VALID_STOPPING_STRATEGIES:
            raise ValueError(
                f"Неизвестная stopping_strategy: '{stopping_strategy}'. "
                f"Допустимые значения: {_VALID_STOPPING_STRATEGIES}."
            )

        self.fetchers = fetchers
        self.probabilities = probabilities
        self.seed = seed
        self.stopping_strategy = stopping_strategy
        self.imbalance_warning_ratio = imbalance_warning_ratio

    def load(self) -> DatasetDict:
        """Загружает все источники и смешивает их train-сплиты.

        Returns:
            DatasetDict с ключом ``'train'`` — смешанным датасетом.

        Raises:
            ValueError: Если один из датасетов не содержит сплита 'train'.
        """
        logger.info(
            "Смешивание %d источников с вероятностями %s (strategy=%s)...",
            len(self.fetchers), self.probabilities, self.stopping_strategy,
        )

        train_splits: list[Dataset] = []

        for i, fetcher in enumerate(self.fetchers):
            ds = fetcher.load()
            if isinstance(ds, DatasetDict):
                if "train" not in ds:
                    raise ValueError(
                        f"Источник #{i} не содержит сплита 'train'. "
                        f"Доступные сплиты: {list(ds.keys())}."
                    )
                train_splits.append(ds["train"])
            else:
                train_splits.append(ds)

            logger.info(
                "  [%d/%d] загружено %d записей (prob=%.2f)",
                i + 1, len(self.fetchers), len(train_splits[-1]), self.probabilities[i],
            )

        # Предупреждаем о сильном дисбалансе размеров при first_exhausted:
        # маленький датасет будет исчерпан раньше и фактически обрежет итоговый объём.
        if self.stopping_strategy == "first_exhausted":
            sizes = [len(s) for s in train_splits]
            min_size, max_size = min(sizes), max(sizes)
            if max_size > min_size * self.imbalance_warning_ratio:
                logger.warning(
                    "Сильный дисбаланс размеров источников: min=%d, max=%d (разница >10x). "
                    "При stopping_strategy='first_exhausted' итоговый датасет будет "
                    "ограничен наименьшим источником. "
                    "Рассмотрите stopping_strategy='all_exhausted' или выравнивание данных.",
                    min_size, max_size,
                )

        mixed_train = interleave_datasets(
            train_splits,
            probabilities=self.probabilities,
            seed=self.seed,
            stopping_strategy=self.stopping_strategy,
        )

        logger.info(
            "Смешивание завершено. Итоговый размер train: %d записей.", len(mixed_train)
        )

        # Оборачиваем в DatasetDict — RandomDatasetSplitter ожидает именно его
        return DatasetDict({"train": mixed_train})