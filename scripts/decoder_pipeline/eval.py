# scripts/eval.py
import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


load_dotenv()
from src.pipelines.base.core.data.builder import DataModule  # noqa
from src.pipelines.decoder.training.module import CausalLMLightningModule  # noqa
from src.utils.checkpoint_utils import load_checkpoint  # noqa
from src.utils.hydra_utils import setup_config  # noqa
from src.utils.logger import setup_logging  # noqa
from src.utils.torch_utils import register_safe_globals  # noqa
from src.tools.storage.resolver import ArtifactResolver  # noqa

setup_logging()
logger = logging.getLogger(__name__)


def _check_drift(
    metrics: dict[str, Any], drift_threshold: float, metric_key: str = "test_perplexity"
) -> None:
    # (Функция остается без изменений, она отлично написана)
    primary_metric = metrics.get(metric_key)

    if primary_metric is None:
        logger.warning("Ключ '%s' не найден в результатах.", metric_key)
        return

    logger.info("Метрика %s: %.4f, порог дрифта: %s", metric_key, primary_metric, drift_threshold)
    is_lower_better = metric_key in ["test_loss", "test_perplexity"]

    if (is_lower_better and primary_metric > drift_threshold) or (
        not is_lower_better and primary_metric < drift_threshold
    ):
        logger.error("ДРИФТ (деградация). Выход с кодом 1.")
        sys.exit(1)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    logger.info("Инициализация компонентов для оценки...")
    # 1. Резолвинг артефактов (Энкодер + БД)
    router = hydra.utils.instantiate(cfg.storage_router)
    cache_base = Path(cfg.paths.model_dir) / "decoder_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    manifest_uri = cfg.manifest.uri

    try:
        _, lora_path = resolver.resolve_and_patch(
            cfg, manifest_uri, pipeline_name="decoder_pipeline"
        )
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов RAG: %s", e)
        sys.exit(1)

    # 2. Сборка Энкодера (с уже пропатченными локальными путями)
    tokenizer = hydra.utils.instantiate(cfg.decoder_pipeline.model.tokenizer).build()

    # Отключаем модификаторы — при инференсе не нужны
    OmegaConf.update(cfg, "decoder_pipeline.model.builder.modifiers", None, force_add=True)

    builder = hydra.utils.instantiate(cfg.decoder_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)

    # Навешиваем адаптер явно если lora-режим
    if lora_path:
        from peft import PeftModel

        logger.info("LoRA: загрузка адаптера из '%s'", lora_path)
        base_model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=False)

    model_module = CausalLMLightningModule(
        model=base_model,
        optimizer_cfg=hydra.utils.instantiate(cfg.decoder_pipeline.optimizer),
        scheduler_cfg=hydra.utils.instantiate(cfg.decoder_pipeline.scheduler)
        if "scheduler" in cfg.decoder_pipeline
        else None,
    )
    datamodule = DataModule(data_cfg=cfg.decoder_pipeline.data, tokenizer=tokenizer)
    training = hydra.utils.instantiate(cfg.decoder_pipeline.training)

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info("Загрузка кастомного Lightning-чекпоинта из: %s", ckpt_path)
        register_safe_globals()
        model_module.model = load_checkpoint(model_module.model, ckpt_path, device="cpu")
        ckpt_path = None

    logger.info("Старт процесса оценки...")
    results = training.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)

    if not results:
        logger.warning("training.test() вернул пустые результаты.")
        return

    metrics = results[0]
    metrics_file = cfg.get("metrics_output_path", "metrics.json")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info("Метрики успешно экспортированы в %s", metrics_file)

    drift_threshold = cfg.get("drift_threshold")
    if drift_threshold is not None:
        _check_drift(
            metrics,
            drift_threshold=drift_threshold,
            metric_key=cfg.get("drift_metric_key", "test_perplexity"),
        )


if __name__ == "__main__":
    expected_pipeline = "decoder_pipeline"

    # Ищем, передал ли пользователь аргумент pipeline_name=...
    pipeline_arg_idx = next(
        (i for i, arg in enumerate(sys.argv) if arg.startswith("pipeline_name=")), None
    )

    if pipeline_arg_idx is not None:
        current_pipeline = sys.argv[pipeline_arg_idx].split("=")[1]
        if current_pipeline != expected_pipeline:
            logger.warning(
                "ВНИМАНИЕ! Запущен RAG-скрипт, но передано pipeline_name=%s. "
                "Принудительно переопределяем на '%s' для предотвращения сбоя конфигов Hydra.",
                current_pipeline,
                expected_pipeline,
            )
            sys.argv[pipeline_arg_idx] = f"pipeline_name={expected_pipeline}"
    else:
        # Если аргумент не передан CLI, Hydra возьмет дефолт из main.yaml.
        # Защищаемся от неправильного дефолта, добавляя аргумент явно:
        sys.argv.append(f"pipeline_name={expected_pipeline}")

    evaluate()
