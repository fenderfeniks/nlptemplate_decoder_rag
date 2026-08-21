# src/tools/benchmark/loader.py
"""Загрузка зафиксированного бенчмарка из Storage через манифест."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)


class BenchmarkLoader:
    """Скачивает бенчмарк из Storage и отдаёт локальный путь или HF Dataset."""

    BENCHMARK_FILENAME = "benchmark.jsonl"

    def __init__(
        self,
        router: Any,
        cache_dir: str | Path,
        manifest_uri: str,
        pipeline_name: str,  # ДОБАВЛЕНО: нужно знать чей бенчмарк грузить
    ) -> None:
        self.router = router
        self.cache_dir = Path(cache_dir)
        self.manifest_uri = manifest_uri
        self.pipeline_name = pipeline_name

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> dict:
        try:
            full_manifest = self.router.download_manifest(
                self.manifest_uri, cache_dir=self.cache_dir
            )
            # Извлекаем словарь только для текущего пайплайна
            return full_manifest.get(self.pipeline_name, {})
        except Exception as e:
            logger.warning("Не удалось загрузить манифест (%s): %s", self.manifest_uri, e)
            return {}

    def _local_path(self) -> Path:
        return self.cache_dir / self.BENCHMARK_FILENAME

    def _cache_is_valid(self, expected_size: int | None) -> bool:
        local = self._local_path()
        if not local.exists():
            return False
        if expected_size is None:
            return True
        actual_size = sum(1 for line in local.open(encoding="utf-8") if line.strip())
        if actual_size != expected_size:
            logger.warning(
                "Размер кэша (%d строк) не совпадает с манифестом (%d). Перекачка.",
                actual_size, expected_size,
            )
            return False
        return True

    def _download(self, benchmark_uri: str) -> Path:
        local = self._local_path()
        logger.info("Загрузка бенчмарка из Storage: %s -> %s", benchmark_uri, local)
        return self.router.download_file_from_uri(
            uri=benchmark_uri,
            local_path=local,
        )

    def resolve_local_path(self) -> Path | None:
        manifest = self._load_manifest()
        benchmark_uri = manifest.get("benchmark_uri")

        if not benchmark_uri:
            logger.info("benchmark_uri не найден в манифесте — BenchmarkExclusion отключён.")
            return None

        expected_size = manifest.get("benchmark_size")

        if self._cache_is_valid(expected_size):
            logger.info("Бенчмарк найден в кэше: %s", self._local_path())
            return self._local_path()

        return self._download(benchmark_uri)

    def load_as_dataset(
        self,
        query_column: str = "question",
        answer_column: str = "answer",
        doc_id_column: str = "chunk_id",
    ) -> HFDataset | None:
        local_path = self.resolve_local_path()
        if local_path is None:
            return None

        records = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not records:
            logger.warning("Бенчмарк пустой: %s", local_path)
            return None

        logger.info("Бенчмарк загружен: %d записей из %s", len(records), local_path)
        return HFDataset.from_list(records)