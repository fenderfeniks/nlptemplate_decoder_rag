# src/pipelines/base/core/data/fetcher.py
import logging
import os
from pathlib import Path
from typing import Any
import boto3
from urllib.parse import urlparse

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

_SUPPORTED_SOURCE_TYPES = ("local", "kaggle", "hf", "s3")


def _detect_loader(file_name: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Определяет тип загрузчика HF datasets по расширению файла.

    Корректно обрабатывает glob-паттерны вида ``*.csv`` и ``data.*.parquet``:
    расширение извлекается из последнего сегмента после финальной точки,
    но только если этот сегмент не является glob-символом.

    Args:
        file_name: Имя файла или glob-паттерн (например, ``math_*.parquet``).
        kwargs: Словарь дополнительных аргументов для загрузчика.

    Returns:
        Кортеж (тип_загрузчика, обновлённые_аргументы).

    Raises:
        ValueError: Если расширение файла не поддерживается или не определено.
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
            f"Неподдерживаемое расширение файла: '.{ext}' (из '{file_name}'). "
            f"Поддерживаются: {list(_EXT_TO_LOADER.keys())}"
        )

    if ext == "tsv" and "sep" not in kwargs:
        kwargs = {**kwargs, "sep": "\t"}

    return loader, kwargs


class RawDataFetcher:
    """Универсальный загрузчик сырых данных.

    Поддерживает источники: локальная файловая система (включая glob-паттерны),
    Kaggle, HuggingFace Hub. Автоматически определяет формат по расширению файла
    и кэширует HF-датасеты на диске.

    Для Kaggle-источника проверяет наличие переменных окружения ``KAGGLE_USERNAME``
    и ``KAGGLE_KEY`` при инициализации (fail-fast), а не в момент скачивания.
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
        """
        Args:
            source_type: Тип источника — 'local', 'kaggle' или 'hf'.
            raw_dir: Директория для хранения/чтения сырых данных.
            dataset_name: Имя датасета (обязательно для 'kaggle' и 'hf').
            file_name: Имя файла или glob-паттерн (обязательно для 'local' и 'kaggle').
            token: Токен доступа HuggingFace (только для приватных датасетов).
            **kwargs: Дополнительные параметры, пробрасываемые в ``load_dataset``.

        Raises:
            ValueError: Если передан неизвестный ``source_type``.
            EnvironmentError: Если источник 'kaggle' и переменные окружения
                ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` не установлены.
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

        # Fail-fast для Kaggle: проверяем credentials при инициализации,
        # а не в момент первого скачивания, когда уже потрачено время на setup.
        if self.source_type == "kaggle":
            self._validate_kaggle_env()
        elif self.source_type == "s3":
            self._validate_s3_env()

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
            "s3": self._load_s3,
        }
        return dispatch[self.source_type]()

    # ------------------------------------------------------------------
    # Приватные загрузчики
    # ------------------------------------------------------------------

    def _load_local(self) -> Dataset | DatasetDict:
        if not self.file_name:
            raise ValueError("Для local источника необходимо указать file_name.")

        matched_files = sorted(self.raw_dir.glob(self.file_name))

        if not matched_files:
            raise FileNotFoundError(
                f"Файлы не найдены по пути/шаблону: {self.raw_dir / self.file_name}"
            )

        data_files = [str(p) for p in matched_files]
        loader, kwargs = _detect_loader(self.file_name, dict(self.kwargs))

        if len(data_files) == 1:
            logger.info("Загрузка локального файла: %s (loader=%s)", data_files[0], loader)
        else:
            logger.info(
                "Загрузка %d файлов по шаблону '%s' (loader=%s)",
                len(data_files), self.file_name, loader,
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
            username = os.environ["KAGGLE_USERNAME"]  # Уже проверено в __init__
            logger.info(
                "Скачиваем '%s' с Kaggle (user: %s)...", self.dataset_name, username
            )
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                self.dataset_name, path=str(self.raw_dir), unzip=True
            )
            logger.info("Скачивание с Kaggle завершено.")

        loader, kwargs = _detect_loader(self.file_name, dict(self.kwargs))
        logger.info("Загрузка Kaggle файла: %s (loader=%s)", file_path, loader)
        return load_dataset(loader, data_files=str(file_path), **kwargs)

    def _load_hf(self) -> Dataset | DatasetDict:
        if not self.dataset_name:
            raise ValueError("Для hf источника необходимо указать dataset_name.")

        # Ключ кэша включает имя датасета и параметры загрузки (split, config и т.д.),
        # чтобы разные конфигурации одного датасета не перезаписывали друг друга.
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

    def _load_s3(self) -> Dataset | DatasetDict:
        if not self.dataset_name:
            raise ValueError("Для s3 источника необходимо указать dataset_name в виде S3 URI (s3://bucket/prefix).")

        parsed = urlparse(self.dataset_name)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")

        # Кэшируем в raw_dir/bucket/prefix
        local_path = self.raw_dir / bucket_name / prefix

        if local_path.exists() and any(local_path.iterdir()):
            logger.info("Данные S3 найдены локально: %s. Скачивание пропущено.", local_path)
        else:
            logger.info("Скачиваем из S3 бакета '%s' (префикс '%s')...", bucket_name, prefix)
            local_path.mkdir(parents=True, exist_ok=True)
            
            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    
                    # Сохраняем структуру директорий
                    rel_path = key[len(prefix):].lstrip("/")
                    file_dest = local_path / rel_path
                    file_dest.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket_name, key, str(file_dest))
                    
            logger.info("Скачивание из S3 завершено.")

        # Натравливаем локальный загрузчик
        loader, kwargs = _detect_loader(self.file_name, dict(self.kwargs))
        target_files = str(local_path / self.file_name) if self.file_name else str(local_path)
        
        logger.info("Загрузка S3 данных: %s (loader=%s)", target_files, loader)
        return load_dataset(loader, data_files=target_files, **kwargs)
    
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

    @staticmethod
    def _validate_s3_env() -> None:
        """Проверяет наличие AWS credentials в окружении."""
        missing = [v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.getenv(v)]
        if missing:
            raise EnvironmentError(
                f"Переменные окружения для S3 не установлены: {missing}. "
                "Проверьте настройки доступа AWS."
            )