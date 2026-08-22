# scripts/decoder/build_benchmark.py
"""Построение и заморозка эталонного SFT-бенчмарка."""

import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import hydra
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

BENCHMARK_FILENAME = "benchmark.jsonl" 

@hydra.main(config_path="../../configs", config_name="build_benchmark", version_base="1.3")
def build_benchmark(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("=== Старт создания SFT эталона (Decoder Benchmark) ===")

    storage_client = hydra.utils.instantiate(cfg.system.storage)
    router = hydra.utils.instantiate(cfg.system.storage_router)
    uri_prefix = cfg.system.storage.uri_prefix
    manifest_uri = cfg.system.manifest.uri

    # 1. Загрузка сырых данных через Fetcher
    logger.info("Загрузка исходных данных...")
    data_cfg = cfg.data
    raw_datasets = hydra.utils.instantiate(data_cfg.source).load()
    
    base_ds = raw_datasets["train"] if "train" in raw_datasets else raw_datasets
    
    # 2. Выборка эталона
    benchmark_size = cfg.get("benchmark_size", 500)
    seed = cfg.get("seed", 42)
    
    n_total = len(base_ds)
    if n_total <= benchmark_size:
        logger.warning(
            "Размер датасета (%d) меньше или равен benchmark_size (%d). "
            "Бенчмарк заберет все данные! Уменьшите benchmark_size.", 
            n_total, benchmark_size
        )
        sample_size = n_total
    else:
        sample_size = benchmark_size
        
    logger.info("Случайная выборка %d записей (seed=%d)...", sample_size, seed)
    shuffled_ds = base_ds.shuffle(seed=seed)
    benchmark_ds = shuffled_ds.select(range(sample_size))
    
    # 3. Нормализация формата
    prompt_col = data_cfg.get("prompt_column", "prompt")
    target_col = data_cfg.get("target_column", "target")
    retrieve_col = data_cfg.get("retrieve_column", None)  # опциональный
    separator = data_cfg.get("separator", "")

    records = []
    for item in benchmark_ds:
        record = {
            "prompt": str(item[prompt_col]) + separator,
            "response": str(item[target_col]),
        }
        # Добавляем context только если колонка задана и присутствует в датасете
        if retrieve_col and retrieve_col in benchmark_ds.column_names:
            raw_ctx = item[retrieve_col]
            record["context"] = str(raw_ctx) if raw_ctx else ""
        
        records.append(record)
        
    # 4. Выгрузка в Storage
    remote_benchmark_dir = f"{cfg.pipeline_name}/benchmarks/latest"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        benchmark_file = tmp_path / BENCHMARK_FILENAME
        with open(benchmark_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        logger.info("Выгрузка бенчмарка в Storage: %s", remote_benchmark_dir)
        storage_client.upload(local_dir=tmp_dir, remote_path=remote_benchmark_dir)

    # 5. Обновление единого манифеста
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            global_manifest = router.download_manifest(manifest_uri, cache_dir=tmp_path)
            logger.info("Найден существующий глобальный манифест. Обновляем секцию %s.", cfg.pipeline_name)
        except Exception:
            logger.warning("Существующий манифест не найден. Будет создан новый.")
            global_manifest = {}

        # Инициализируем словарь пайплайна, если его еще нет
        if cfg.pipeline_name not in global_manifest:
            global_manifest[cfg.pipeline_name] = {}

        global_manifest[cfg.pipeline_name]["benchmark_uri"] = f"{uri_prefix}{remote_benchmark_dir}/{BENCHMARK_FILENAME}"
        global_manifest[cfg.pipeline_name]["benchmark_updated_at"] = datetime.now(timezone.utc).isoformat()
        global_manifest[cfg.pipeline_name]["benchmark_size"] = len(records)

        manifest_file = tmp_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(global_manifest, f, indent=4, ensure_ascii=False)

        # Безопасно загружаем ТОЛЬКО один файл
        storage_client.upload_file(local_path=manifest_file, remote_path="manifest.json")

    logger.info(
        "=== SFT Бенчмарк успешно зафиксирован. URI: %s%s/%s ===",
        uri_prefix, remote_benchmark_dir, BENCHMARK_FILENAME
    )

if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline
    enforce_pipeline("decoder_pipeline")
    build_benchmark()