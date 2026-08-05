# scripts/rag/eval.py
import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.pipelines.base.core.data.builder import DataModule
from src.pipelines.rag.training.module import RAGLightningModule
from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    """Оценка качества RAG-энкодера на отложенной выборке.

    Использует ``training.validate()``, который вызывает ``on_validation_epoch_end``
    у зарегистрированных callbacks — в том числе ``RetrievalEvaluationCallback``,
    считающий MRR@K, Recall@K и NDCG@K.

    Ожидает конфиг с ``rag_pipeline.data.task='contrastive'`` и
    ``rag_pipeline.training.callbacks`` содержащим ``RetrievalEvaluationCallback``.
    """
    cfg = setup_config(cfg)
    logger.info("Старт оценки RAG-энкодера (Retrieval Evaluation)...")

    pl.seed_everything(cfg.seed, workers=True)

    # 1. Резолвинг артефактов (Энкодер + БД)
    router = hydra.utils.instantiate(cfg.storage_router)
    cache_base = Path(cfg.paths.model_dir) / "rag_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    manifest_uri = cfg.manifest.uri

    try:
        db_dir, lora_path = resolver.resolve_and_patch(
            cfg, manifest_uri, pipeline_name="rag_pipeline"
        )
        if not db_dir:
            raise ValueError("Манифест не содержит 'vector_db_uri'. База не найдена.")
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов RAG: %s", e)
        sys.exit(1)

    # 2. Сборка Энкодера (с уже пропатченными локальными путями)
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()

    # Отключаем модификаторы — при инференсе не нужны
    OmegaConf.update(cfg, "rag_pipeline.model.builder.modifiers", None, force_add=True)

    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)

    # Навешиваем адаптер явно если lora-режим
    if lora_path:
        from peft import PeftModel

        logger.info("LoRA: загрузка адаптера из '%s'", lora_path)
        base_model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=False)

    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    # Loss нужен для инициализации RAGLightningModule (интерфейс требует),
    # но при training.validate() backward не вызывается — Loss не считается.
    loss_fn = hydra.utils.instantiate(cfg.rag_pipeline.loss)

    # 2. DataModule в режиме contrastive
    # Для оценки используем val_dataloader: RetrievalEvaluationCallback
    # привязан к on_validation_epoch_end и вызывает training.datamodule.val_dataloader().
    datamodule = DataModule(data_cfg=cfg.rag_pipeline.data, tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage="validate")

    val_loader = datamodule.val_dataloader()
    if val_loader is None:
        logger.error(
            "val_dataloader пуст — оценка невозможна. "
            "Проверьте val_size в конфиге (должен быть > 0 для contrastive)."
        )
        return

    # 3. LightningModule
    model_module = RAGLightningModule(
        model=base_model,
        pooler=pooler,
        loss_fn=loss_fn,
        optimizer_cfg=hydra.utils.instantiate(cfg.rag_pipeline.optimizer),
        scheduler_cfg=None,  # При оценке планировщик не нужен
    )

    # 4. training (должен содержать RetrievalEvaluationCallback в конфиге)
    training = hydra.utils.instantiate(cfg.rag_pipeline.training)

    # 5. Запуск оценки
    # training.validate() прогоняет один validation epoch и вызывает все callbacks.
    # Метрики логируются через pl_module.log() и попадают в MLflow/WandB.
    logger.info("Запуск training.validate()...")
    training.validate(model=model_module, datamodule=datamodule)
    logger.info("Оценка завершена.")


if __name__ == "__main__":
    expected_pipeline = "rag_pipeline"

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
