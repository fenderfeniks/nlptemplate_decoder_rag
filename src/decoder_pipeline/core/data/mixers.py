# src/decoder_pipeline/core/data/mixers.py
import logging
from typing import Any

from datasets import Dataset, DatasetDict, interleave_datasets

logger = logging.getLogger(__name__)

_VALID_STOPPING_STRATEGIES = ("first_exhausted", "all_exhausted")

class InterleavedDataFetcher:
    """Класс для смешивания нескольких источников данных.

    Выступает прозрачной заменой для RawDataFetcher. Загружает список
    фетчеров и смешивает их данные с заданными вероятностями.
    """

    def __init__(
        self,
        fetchers: list[Any],
        probabilities: list[float],
        seed: int = 42,
        stopping_strategy: str = "first_exhausted",
        imbalance_warning_ratio: float = 10.0,
    ) -> None:
        """Инициализирует миксер датасетов.

        Args:
            fetchers: Список инстанцированных объектов-загрузчиков.
            probabilities: Список вероятностей выборки из каждого датасета 
                (сумма должна быть равна 1.0).
            seed: Seed для воспроизводимого смешивания.
            stopping_strategy: 'first_exhausted' (остановка, когда закончится 
                самый маленький датасет) или 'all_exhausted' (когда закончатся все).
        """
        if len(fetchers) != len(probabilities):
            raise ValueError(
                "Количество fetchers должно совпадать с количеством probabilities."
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
        """Загружает и смешивает датасеты.

        Returns:
            DatasetDict, содержащий смешанный сплит 'train'.
        """
        logger.info(
            "Смешивание %d датасетов с вероятностями %s...", 
            len(self.fetchers), self.probabilities
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
        
        logger.info("Смешивание завершено. Итоговый размер train: %d записей", len(mixed_train))
        
        # Оборачиваем в DatasetDict, чтобы сплиттер (RandomDatasetSplitter)
        # позже смог нарезать из этого валидацию и тест
        return DatasetDict({"train": mixed_train})