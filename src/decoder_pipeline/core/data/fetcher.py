# src/decoder_pipeline/core/data/fetcher.py
import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)

# Маппинг расширений -> тип загрузчика HF datasets
_EXT_TO_LOADER: dict[str, str] = {
    "csv": "csv",
    "tsv": "csv",  # load_dataset("csv", sep="\t")
    "json": "json",
    "jsonl": "json",
    "txt": "text",
    "parquet": "parquet",
    "pq": "parquet",
    "arrow": "arrow",
}

_SUPPORTED_SOURCE_TYPES = ("local", "kaggle", "hf")

def _detect_loader(file_name: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Определяет тип загрузчика HF datasets по расширению файла.

    Для файлов формата TSV автоматически добавляет параметр sep='\\t' 
    в аргументы загрузчика.

    Args:
        file_name: Имя файла с данными.
        kwargs: Словарь дополнительных аргументов для загрузчика.

    Returns:
        Кортеж (тип_загрузчика, обновленные_аргументы).

    Raises:
        ValueError: Если расширение файла не поддерживается.
    """
    parts = file_name.rsplit(".", 1)
    if len(parts) < 2:
        raise ValueError(
            f"Не удалось определить расширение файла из '{file_name}'. "
            f"Поддерживаются: {list(_EXT_TO_LOADER.keys())}"
        )

    raw_ext = parts[-1].lower()

    ext = raw_ext.strip("*?[]")
    loader = _EXT_TO_LOADER.get(ext)

    if loader is None:
        raise ValueError(
            f"Неподдерживаемое расширение файла: '.{ext}'. "
            f"Поддерживаются: {list(_EXT_TO_LOADER.keys())}"
        )

    if ext == "tsv" and "sep" not in kwargs:
        kwargs = {**kwargs, "sep": "\t"}

    return loader, kwargs


class RawDataFetcher:
    """Универсальный класс для получения сырых данных.

    Проверяет локальное наличие, при необходимости скачивает с Kaggle 
    или HuggingFace. Автоматически определяет формат файла по расширению.
    """

    def __init__(
        self,
        source_type: str,
        raw_dir: str | Path,
        dataset_name: str | None = None,
        file_name: str | None = None,
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Инициализирует загрузчик данных.

        Args:
            source_type: Тип источника данных ('local', 'kaggle', 'hf').
            raw_dir: Директория для сохранения/чтения сырых данных.
            dataset_name: Имя датасета (для Kaggle или HF).
            file_name: Имя конкретного файла для загрузки.
            token: Токен доступа (используется только для HuggingFace).
            **kwargs: Дополнительные параметры для `load_dataset`.
        """
        if source_type not in _SUPPORTED_SOURCE_TYPES:
            raise ValueError(
                f"Неизвестный тип источника данных: '{source_type}'. "
                f"Поддерживаются: {_SUPPORTED_SOURCE_TYPES}."
            )
        
        self.source_type = source_type
        self.raw_dir = Path(raw_dir)
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.token = token
        self.kwargs = kwargs

        if self.source_type == "kaggle":
                    self._validate_kaggle_env()

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def load(self) -> Dataset | DatasetDict:
        """Загружает датасет из сконфигурированного источника.

        Returns:
            Dataset или DatasetDict в зависимости от источника и формата файла.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        dispatch = {
            "local": self._load_local,
            "kaggle": self._load_kaggle,
            "hf": self._load_hf,
        }
        return dispatch[self.source_type]()

    # ------------------------------------------------------------------
    # Приватные загрузчики
    # ------------------------------------------------------------------

    def _load_local(self) -> Dataset | DatasetDict:
        if not self.file_name:
            raise ValueError("Для local источника необходимо указать file_name.")
            
        matched_files = list(self.raw_dir.glob(self.file_name))
        
        if not matched_files:
            raise FileNotFoundError(
                f"Локальные файлы не найдены по пути или шаблону: {self.raw_dir / self.file_name}"
            )

        data_files = [str(p) for p in matched_files]
        loader, kwargs = _detect_loader(self.file_name, self.kwargs)
        
        if len(data_files) == 1:
            logger.info("Загрузка локального файла: %s (loader=%s)", data_files[0], loader)
        else:
            logger.info(
                "Загрузка %d локальных файлов по шаблону %s (loader=%s)", 
                len(data_files), self.file_name, loader
            )
            
        return load_dataset(loader, data_files=data_files, **kwargs)

    def _load_kaggle(self) -> Dataset | DatasetDict:
        if not self.file_name or not self.dataset_name:
            raise ValueError("Для kaggle источника необходимы dataset_name и file_name.")

        # Импортируем здесь: kaggle — опциональная зависимость
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise ImportError(
                "Для kaggle источника установите: pip install kaggle"
            ) from e
        
        file_path = self.raw_dir / self.file_name

        if file_path.exists():
            logger.info(
                "Kaggle датасет найден локально: %s. Скачивание пропущено.", file_path
            )
        else:
            username = os.getenv("KAGGLE_USERNAME")
            logger.info("Скачиваем %s с Kaggle (user: %s)...", self.dataset_name, username)
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(self.dataset_name, path=str(self.raw_dir), unzip=True)
            logger.info("Скачивание с Kaggle завершено.")

        loader, kwargs = _detect_loader(self.file_name, self.kwargs)
        logger.info("Загрузка Kaggle файла: %s (loader=%s)", file_path, loader)
        return load_dataset(loader, data_files=str(file_path), **kwargs)

    def _load_hf(self) -> Dataset | DatasetDict:
        if not self.dataset_name:
            raise ValueError("Для hf источника необходимо указать dataset_name.")

        cache_key = self.dataset_name.replace("/", "_")
        extra = {k: v for k, v in self.kwargs.items() if k in ("name", "split")}
        if extra:
            suffix = "_".join(f"{k}-{v}" for k, v in sorted(extra.items()))
            cache_key = f"{cache_key}_{suffix}"

        hf_local_path = self.raw_dir / cache_key

        if hf_local_path.exists():
            logger.info("HF датасет найден в кэше: %s", hf_local_path)
            return load_from_disk(str(hf_local_path))

        logger.info("Скачиваем '%s' из HuggingFace Hub...", self.dataset_name)
        dataset = load_dataset(self.dataset_name, token=self.token, **self.kwargs)
        dataset.save_to_disk(str(hf_local_path))
        logger.info("HF датасет сохранён в кэш: %s", hf_local_path)
        return dataset

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_kaggle_env() -> None:
        """Проверяет наличие Kaggle credentials в окружении (fail-fast)."""
        missing = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(v)]
        if missing:
            raise EnvironmentError(
                f"Переменные окружения не установлены: {missing}. "
                "Для K8s: проверь Secret и прокидывание через KubernetesPodOperator env."
            )