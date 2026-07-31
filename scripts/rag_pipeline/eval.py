# scripts/rag/eval.py
import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.rag_pipeline.core.data.builder import RAGDataModule
from src.rag_pipeline.training.module import RAGLightningModule
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

    # 1. Токенизатор и энкодер
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)

    # При наличии LoRA — передаём путь к адаптеру перед build().
    # Для eval адаптер загружается в режиме is_trainable=False (inference).
    lora_resume = cfg.rag_pipeline.model.get("lora_resume_path", None)
    if lora_resume:
        builder.lora_resume_path = lora_resume
        logger.info("LoRA: загрузка адаптера из '%s'", lora_resume)

    base_model = builder.build(tokenizer=tokenizer)
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    # Loss нужен для инициализации RAGLightningModule (интерфейс требует),
    # но при training.validate() backward не вызывается — Loss не считается.
    loss_fn = hydra.utils.instantiate(cfg.rag_pipeline.loss)

    # 2. DataModule в режиме contrastive
    # Для оценки используем val_dataloader: RetrievalEvaluationCallback
    # привязан к on_validation_epoch_end и вызывает training.datamodule.val_dataloader().
    datamodule = RAGDataModule(data_cfg=cfg.rag_pipeline.data, tokenizer=tokenizer)
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
    evaluate()
