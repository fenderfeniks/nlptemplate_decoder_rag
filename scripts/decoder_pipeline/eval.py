# scripts/decoder/eval.py
"""Оценка качества декодер-модели на тестовой выборке."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig


load_dotenv()

from src.pipelines.base.core.data.builder import DataModule  # noqa: E402
from src.pipelines.decoder.inference.builder import build_decoder_model  # noqa: E402
from src.pipelines.decoder.training.module import CausalLMLightningModule  # noqa: E402
from src.tools.storage.resolver import ArtifactResolver  # noqa: E402
from src.utils.checkpoint_utils import load_checkpoint  # noqa: E402
from src.utils.hydra_utils import setup_config  # noqa: E402
from src.utils.logger import setup_logging  # noqa: E402
from src.utils.torch_utils import register_safe_globals  # noqa: E402


setup_logging()
logger = logging.getLogger(__name__)


def _check_drift(
    metrics: dict[str, Any],
    drift_threshold: float,
    metric_key: str = "test_perplexity",
) -> None:
    """Проверить метрику на деградацию относительно порога.

    Завершает процесс с кодом 1 если метрика хуже порога.
    is_lower_better=True для loss/perplexity, False для accuracy/f1.
    """
    primary_metric = metrics.get(metric_key)

    if primary_metric is None:
        logger.warning("Ключ '%s' не найден в результатах.", metric_key)
        return

    logger.info("Метрика %s: %.4f, порог дрифта: %s", metric_key, primary_metric, drift_threshold)
    is_lower_better = metric_key in ("test_loss", "test_perplexity")

    if (is_lower_better and primary_metric > drift_threshold) or (
        not is_lower_better and primary_metric < drift_threshold
    ):
        logger.error("ДРИФТ (деградация). Выход с кодом 1.")
        sys.exit(1)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    """Оценка декодер-модели: trainer.test() + экспорт метрик + drift-check."""
    cfg = setup_config(cfg)
    logger.info("Инициализация компонентов для оценки...")

    # 1. Резолвинг артефактов
    router = hydra.utils.instantiate(cfg.storage_router)
    cache_base = Path(cfg.paths.model_dir) / "decoder_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    try:
        _, lora_path = resolver.resolve_and_patch(
            cfg, cfg.manifest.uri, pipeline_name="decoder_pipeline"
        )
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов: %s", e)
        sys.exit(1)

    # 2. Сборка модели
    base_model, tokenizer = build_decoder_model(cfg, lora_path)

    # 3. LightningModule, DataModule, Trainer
    model_module = CausalLMLightningModule(
        model=base_model,
        optimizer_cfg=hydra.utils.instantiate(cfg.decoder_pipeline.optimizer),
        scheduler_cfg=hydra.utils.instantiate(cfg.decoder_pipeline.scheduler)
        if "scheduler" in cfg.decoder_pipeline
        else None,
    )
    datamodule = DataModule(data_cfg=cfg.decoder_pipeline.data, tokenizer=tokenizer)
    trainer = hydra.utils.instantiate(cfg.decoder_pipeline.training)

    # 4. Опциональная загрузка кастомного чекпоинта
    # ckpt_path=None → trainer.test() использует best_model_path из ModelCheckpoint.
    # ckpt_path задан → грузим веса вручную, передаём ckpt_path=None в test().
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info("Загрузка кастомного Lightning-чекпоинта из: %s", ckpt_path)
        register_safe_globals()
        model_module.model = load_checkpoint(model_module.model, ckpt_path, device="cpu")
        ckpt_path = None

    # 5. Тест и экспорт метрик
    logger.info("Старт процесса оценки...")
    results = trainer.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)

    if not results:
        logger.warning("trainer.test() вернул пустые результаты.")
        return

    metrics = results[0]
    metrics_file = cfg.get("metrics_output_path", "metrics.json")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info("Метрики успешно экспортированы в %s", metrics_file)

    # 6. Drift-check (опционально — только если задан порог)
    drift_threshold = cfg.get("drift_threshold")
    if drift_threshold is not None:
        _check_drift(
            metrics,
            drift_threshold=drift_threshold,
            metric_key=cfg.get("drift_metric_key", "test_perplexity"),
        )


if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline

    enforce_pipeline("decoder_pipeline")
    evaluate()
