# src/pipelines/base/core/data/transforms/exclusion.py
"""Исключает из обучающей выборки примеры, попавшие в эталонный бенчмарк.

Предотвращает утечку данных (data leakage). Поддерживает два режима:

1. Статический путь (benchmark_path):
   Путь задан явно в конфиге. Используется если бенчмарк уже скачан
   и лежит локально (например в CI после `benchmark_loader.resolve_local_path()`).

2. Динамическая загрузка через BenchmarkLoader:
   BenchmarkLoader скачивает бенчмарк из манифеста при первом вызове.
   Используется в продакшн-DataModule где путь неизвестен заранее.

В конфиге transforms указывается `benchmark_path` — он заполняется
ArtifactResolver'ом или BenchmarkLoader'ом перед запуском DataModule.
Сам трансформ не знает об Storage — это не его ответственность.

Почему MD5, а не SHA-256:
    Для хэширования контента с целью дедупликации MD5 достаточен.
    Коллизии на масштабе датасета (<10M записей) практически исключены.
    Криптографическая стойкость не нужна — нужна скорость и компактность.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from src.pipelines.base.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class BenchmarkExclusionTransform(BaseDatasetTransform):
    """Фильтрует обучающую выборку по MD5-хэшам записей из бенчмарка.

    Args:
        benchmark_path:   Путь к JSONL-файлу бенчмарка. Если None — трансформ
                          пробует получить путь через benchmark_loader.
                          Если оба не заданы — фильтрация пропускается с WARNING.
        benchmark_loader: Опциональный BenchmarkLoader. Используется если
                          benchmark_path не задан: loader.resolve_local_path()
                          скачивает бенчмарк из манифеста и возвращает путь.
        dataset_columns:  Колонки датасета для хэширования (напр. ["text"]).
        benchmark_columns: Колонки бенчмарка для хэширования (напр. ["context"]).
                          Если None — используются те же имена что в dataset_columns.
        column_separator: Разделитель при конкатенации нескольких колонок.
        num_proc:         Число процессов для параллельного фильтра.
    """

    def __init__(
        self,
        dataset_columns: list[str],
        benchmark_path: str | None = None,
        benchmark_loader: Any | None = None,
        benchmark_columns: list[str] | None = None,
        column_separator: str = "\n\n",
        num_proc: int = 4,
    ) -> None:
        if not dataset_columns:
            raise ValueError("dataset_columns не может быть пустым.")
        if benchmark_path is None and benchmark_loader is None:
            raise ValueError(
                "Необходимо передать benchmark_path или benchmark_loader. "
                "benchmark_path берётся из конфига после резолвинга манифеста, "
                "benchmark_loader — из DataModule при создании."
            )

        self.dataset_columns = dataset_columns
        self.benchmark_columns = benchmark_columns or dataset_columns
        self.column_separator = column_separator
        self.num_proc = num_proc

        # Резолвинг пути: явный путь имеет приоритет над loader
        if benchmark_path is not None:
            self._benchmark_path: Path | None = Path(benchmark_path)
        else:
            # Ленивая загрузка — вызываем loader только при первом __call__
            self._benchmark_path = None
            self._benchmark_loader = benchmark_loader

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _resolve_path(self) -> Path | None:
        """Возвращает локальный путь к бенчмарку, скачивая если нужно."""
        if self._benchmark_path is not None:
            return self._benchmark_path

        # Ленивая загрузка через loader
        loader = getattr(self, "_benchmark_loader", None)
        if loader is None:
            return None

        resolved = loader.resolve_local_path()
        if resolved is not None:
            # Кэшируем чтобы не скачивать повторно если трансформ применяется к нескольким сплитам
            self._benchmark_path = resolved
        return resolved

    def _get_benchmark_hashes(self) -> set[str]:
        """Читает JSONL и возвращает множество MD5-хэшей по benchmark_columns."""
        path = self._resolve_path()

        if path is None:
            logger.warning(
                "Путь к бенчмарку не удалось определить. "
                "Фильтрация data leakage пропущена."
            )
            return set()

        if not path.exists():
            logger.warning(
                "Бенчмарк не найден по пути %s. Фильтрация пропущена.", path
            )
            return set()

        hashes: set[str] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    combined = self.column_separator.join(
                        str(record.get(c, "")) for c in self.benchmark_columns
                    )
                    hashes.add(hashlib.md5(combined.encode("utf-8")).hexdigest())
                except json.JSONDecodeError:
                    continue

        logger.info(
            "Считано %d уникальных хэшей из бенчмарка (%s) для exclusion list.",
            len(hashes), path,
        )
        return hashes

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def __call__(self, dataset: HFDataset) -> HFDataset:
        active_cols = [c for c in self.dataset_columns if c in dataset.column_names]
        if not active_cols:
            logger.warning(
                "Колонки %s не найдены в датасете (доступные: %s). "
                "Исключение эталона пропущено.",
                self.dataset_columns, dataset.column_names,
            )
            return dataset

        benchmark_hashes = self._get_benchmark_hashes()
        if not benchmark_hashes:
            return dataset

        initial_count = len(dataset)
        logger.info(
            "Запуск фильтрации data leakage (колонки: %s, хэшей в blacklist: %d)...",
            active_cols, len(benchmark_hashes),
        )

        # Выносим всё из self в локальные переменные — dill не будет
        # пытаться сериализовать StorageRouter/SSLContext
        _hashes = benchmark_hashes        # set[str] — пикклируется
        _cols = active_cols               # list[str] — пикклируется
        _sep = self.column_separator      # str — пикклируется

        def _is_not_in_benchmark(example: dict[str, Any]) -> bool:
            combined = _sep.join(str(example[c]) for c in _cols)
            return hashlib.md5(combined.encode("utf-8")).hexdigest() not in _hashes

        filtered_ds = dataset.filter(
            _is_not_in_benchmark,
            num_proc=self.num_proc,
            desc="Excluding benchmark records",
        )

        removed = initial_count - len(filtered_ds)
        logger.info(
            "Фильтрация завершена: удалено %d записей (%.1f%%). Осталось: %d.",
            removed, (removed / initial_count * 100) if initial_count else 0, len(filtered_ds),
        )

        return filtered_ds