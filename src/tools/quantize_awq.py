# scripts/tools/quantize_awq.py
"""Офлайн AWQ-квантизация модели из манифеста.

Логика идентична merge_and_export (scripts/tools/merge_lora.py):
    1. Резолвинг артефактов из манифеста через ArtifactResolver
    2. Загрузка merged fp16 модели (load_type: full_model)
    3. Калибровка и квантизация через AutoAWQ на данных бенчмарка
    4. Сохранение локально → загрузка в Storage
    5. Обновление манифеста: model_uri → AWQ-версия

Запускается один раз после merge_lora, результат кладётся в storage.
Инференс после этого использует quantization: awq в конфиге.

Требования:
    pip install autoawq
    CUDA обязательна — AWQ калибровка не работает на CPU.
"""

import gc
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.tools.benchmark.loader import BenchmarkLoader
from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


def _load_calibration_data(
    cfg: DictConfig,
    router: Any,
    cache_base: Path,
    query_column: str | None = None,
    n_samples: int = 128,
) -> list[str]:
    """Загружает калибровочные данные из бенчмарка пайплайна.

    Использует тот же BenchmarkLoader что и eval.py — единый источник данных.
    AWQ калибруется на 128 сэмплах — больше не нужно, качество не растёт.

    Args:
        n_samples: Количество сэмплов для калибровки. 128 — стандарт для AWQ.
    """
    benchmark_loader = BenchmarkLoader(
        router=router,
        cache_dir=cache_base / "benchmark",
        manifest_uri=cfg.system.manifest.uri,
        pipeline_name=cfg.pipeline_name,
    )

    load_kwargs = {}
    if query_column:
        load_kwargs["query_column"] = query_column

    dataset = benchmark_loader.load_as_dataset(**load_kwargs)

    if dataset is None or len(dataset) == 0:
        logger.warning(
            "Бенчмарк не найден для '%s'. "
            "Калибровка на пустых данных деградирует качество AWQ. "
            "Запусти build_benchmark.py перед квантизацией.",
            cfg.pipeline_name,
        )
        return []

    # Определяем колонку с текстом — те же fallbacks что в endpoints/eval.py
    available = list(dataset.column_names)
    text_col = query_column
    if not text_col:
        for candidate in ["question", "query", "prompt", "input", "text"]:
            if candidate in available:
                text_col = candidate
                break

    if not text_col:
        logger.warning(
            "Не найдена текстовая колонка в бенчмарке. Доступные: %s. "
            "Используем пустую калибровку.",
            available,
        )
        return []

    texts = [str(item[text_col]) for item in dataset]
    sampled = texts[:n_samples]
    logger.info(
        "Калибровочные данные: %d сэмплов из бенчмарка (колонка '%s').",
        len(sampled), text_col,
    )
    return sampled


@hydra.main(config_path="../../configs", config_name="promote", version_base="1.3")
def quantize_and_export(cfg: DictConfig) -> None:
    """Квантизует модель из манифеста через AWQ и экспортирует в хранилище."""
    cfg = setup_config(cfg)

    if not torch.cuda.is_available():
        logger.critical(
            "CUDA недоступна. AWQ калибровка требует GPU. "
            "Запусти скрипт на машине с CUDA."
        )
        sys.exit(1)

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        logger.critical(
            "Пакет autoawq не установлен. "
            "Установи: pip install autoawq"
        )
        sys.exit(1)

    storage_client = hydra.utils.instantiate(cfg.system.storage)
    router = hydra.utils.instantiate(cfg.system.storage_router)
    uri_prefix = cfg.system.storage.uri_prefix
    manifest_uri = cfg.system.manifest.uri

    mlflow_model_name = cfg.model.architecture.get("mlflow_model_name")
    if not mlflow_model_name:
        raise ValueError(
            f"mlflow_model_name не задан в model.architecture для '{cfg.pipeline_name}'"
        )

    # AWQ-параметры из конфига (configs/quantization/awq_export.yaml)
    # или дефолты если не заданы
    awq_cfg = cfg.get("awq", {})
    w_bit: int = awq_cfg.get("w_bit", 4)
    q_group_size: int = awq_cfg.get("q_group_size", 128)
    zero_point: bool = awq_cfg.get("zero_point", True)
    # GEMM быстрее на батчах, GEMV — на одиночных запросах (batch_size=1)
    version: str = awq_cfg.get("version", "GEMM")
    n_calib_samples: int = awq_cfg.get("n_calib_samples", 128)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": version,
    }

    # Имя в storage: включаем битность чтобы различать 4bit vs 8bit версии
    remote_awq_dir = f"awq_models/{mlflow_model_name}_w{w_bit}g{q_group_size}"

    logger.info(
        "AWQ квантизация: модель='%s', w_bit=%d, group_size=%d, version=%s",
        mlflow_model_name, w_bit, q_group_size, version,
    )

    # ── 1. Проверка: уже квантизовано? ───────────────────────────────────
    if storage_client.exists(remote_awq_dir):
        logger.info(
            "AWQ модель уже существует в '%s'. "
            "Квантизация пропущена. Обновляем только манифест.",
            remote_awq_dir,
        )
    else:
        logger.info("AWQ модель не найдена. Начинаем квантизацию...")

        # ── 2. Резолвинг модели из манифеста ─────────────────────────────
        cache_base = Path(cfg.system.paths.model_dir) / f"{cfg.pipeline_name}_cache"
        resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

        try:
            _, lora_path, _ = resolver.resolve_and_patch(
                cfg, manifest_uri,
                pipeline_name=cfg.pipeline_name,
                is_training=False,
            )
        except Exception as e:
            logger.critical("Сбой резолвинга артефактов: %s", e)
            sys.exit(1)

        if lora_path:
            logger.error(
                "Манифест указывает на LoRA адаптер, а не на merged модель. "
                "Сначала запусти merge_lora.py чтобы получить монолитную fp16 модель, "
                "затем повтори квантизацию."
            )
            sys.exit(1)

        # model_name_or_path уже пропатчен резолвером в cfg.model.builder
        model_path = cfg.model.builder.model_name_or_path
        logger.info("Загрузка fp16 модели для квантизации из: %s", model_path)

        # ── 3. Загрузка калибровочных данных из бенчмарка ────────────────
        data_cfg = cfg.get("data", {})
        calib_texts = _load_calibration_data(
            cfg=cfg,
            router=router,
            cache_base=cache_base,
            query_column=data_cfg.get("query_column"),
            n_samples=n_calib_samples,
        )

        # ── 4. Загрузка модели через AutoAWQ ─────────────────────────────
        # AutoAWQ использует собственный from_pretrained — не HFModelBuilder.
        # Это нормально: квантизация — офлайн инструмент, не часть пайплайна.
        tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # ── 5. Калибровка и квантизация ───────────────────────────────────
        logger.info("Запуск AWQ калибровки (%d сэмплов)...", len(calib_texts))
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_texts if calib_texts else None,
        )
        logger.info("AWQ квантизация завершена.")

        # ── 6. Локальное сохранение ───────────────────────────────────────
        local_awq_path = (
            Path(cfg.system.paths.model_dir)
            / f"awq_{mlflow_model_name}_w{w_bit}g{q_group_size}"
        )
        local_awq_path.mkdir(parents=True, exist_ok=True)
        logger.info("Сохранение AWQ модели локально: %s", local_awq_path)
        model.save_quantized(local_awq_path)
        tokenizer.save_pretrained(local_awq_path)

        # ── 7. Загрузка в Storage ─────────────────────────────────────────
        logger.info("Выгрузка AWQ модели в Storage: %s", remote_awq_dir)
        storage_client.upload(local_dir=local_awq_path, remote_path=remote_awq_dir)

        # ── 8. Очистка памяти ─────────────────────────────────────────────
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Память GPU освобождена.")

    # ── 9. Обновление манифеста ───────────────────────────────────────────
    # Та же логика что в merge_lora: точечное обновление секции пайплайна.
    pipeline_name = cfg.pipeline_name
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        old_manifest_dir = tmp_path / "old_manifest"

        try:
            manifest = router.download_manifest(manifest_uri, cache_dir=old_manifest_dir)
            logger.info("Манифест найден. Обновляем секцию '%s'.", pipeline_name)
        except Exception:
            logger.warning("Манифест не найден. Будет создан новый.")
            manifest = {}

        if pipeline_name not in manifest:
            manifest[pipeline_name] = {}

        manifest[pipeline_name].update({
            "load_type": "full_model",
            "model_uri": f"{uri_prefix}{remote_awq_dir}",
            # Сохраняем метаданные квантизации — полезно для аудита
            "quantization": {
                "method": "awq",
                "w_bit": w_bit,
                "q_group_size": q_group_size,
                "version": version,
                "n_calib_samples": n_calib_samples,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        # lora_uri больше не актуален после квантизации монолита
        manifest[pipeline_name].pop("lora_uri", None)
        manifest[pipeline_name].pop("base_model_uri", None)

        manifest_file = tmp_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload_file(local_path=manifest_file, remote_path="manifest.json")

    logger.info(
        "Манифест обновлён для '%s'. "
        "Для инференса используй quantization: awq в конфиге.",
        pipeline_name,
    )


if __name__ == "__main__":
    quantize_and_export()