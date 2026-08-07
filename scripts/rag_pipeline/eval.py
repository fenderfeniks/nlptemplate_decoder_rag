# scripts/rag/eval.py
import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.pipelines.base.core.data.builder import DataModule
from src.pipelines.rag.inference.builder import build_inference_encoder
from src.pipelines.rag.training.module import RAGLightningModule
from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    """Оценка качества RAG-энкодера на отложенной выборке.

    Использует ``trainer.validate()``, который вызывает ``on_validation_epoch_end``
    у зарегистрированных callbacks — в том числе ``RetrievalEvaluationCallback``,
    считающий MRR@K, Recall@K и NDCG@K.

    Ожидает конфиг с ``rag_pipeline.data.task='contrastive'`` и
    ``rag_pipeline.training.callbacks`` содержащим ``RetrievalEvaluationCallback``.
    """
    cfg = setup_config(cfg)
    logger.info("Старт оценки RAG-энкодера (Retrieval Evaluation)...")

    pl.seed_everything(cfg.seed, workers=True)

    # 1. Резолвинг артефактов (энкодер + БД)
    router = hydra.utils.instantiate(cfg.storage_router)
    cache_base = Path(cfg.paths.model_dir) / "rag_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    try:
        db_dir, lora_path = resolver.resolve_and_patch(
            cfg, cfg.manifest.uri, pipeline_name="rag_pipeline"
        )
        if not db_dir:
            raise ValueError("Манифест не содержит 'vector_db_uri'. База не найдена.")
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов RAG: %s", e)
        sys.exit(1)

    # 2. Сборка энкодера
    base_model, pooler, tokenizer = build_inference_encoder(cfg, lora_path)

    # 3. DataModule в режиме contrastive (val_dataloader → RetrievalEvaluationCallback)
    datamodule = DataModule(data_cfg=cfg.rag_pipeline.data, tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage="validate")

    if datamodule.val_dataloader() is None:
        logger.error(
            "val_dataloader пуст — оценка невозможна. "
            "Проверьте val_size в конфиге (должен быть > 0 для contrastive)."
        )
        return

    # 4. LightningModule
    # Loss нужен для интерфейса, но при validate() backward не вызывается.
    model_module = RAGLightningModule(
        model=base_model,
        pooler=pooler,
        loss_fn=hydra.utils.instantiate(cfg.rag_pipeline.loss),
        optimizer_cfg=hydra.utils.instantiate(cfg.rag_pipeline.optimizer),
        scheduler_cfg=None,
    )

    # 5. Запуск оценки через Trainer (несёт RetrievalEvaluationCallback из конфига)
    logger.info("Запуск trainer.validate()...")
    trainer = hydra.utils.instantiate(cfg.rag_pipeline.training)
    trainer.validate(model=model_module, datamodule=datamodule)
    logger.info("Оценка завершена.")


if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline

    enforce_pipeline("rag_pipeline")
    evaluate()
