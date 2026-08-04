# scripts/rag_pipeline/index_db.py
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.pipelines.base.core.data.builder import DataModule
from src.pipelines.rag.indexing.indexer import KnowledgeBaseIndexer
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def index_database(cfg: DictConfig) -> None:
    """Оффлайн-индексация корпуса документов в FAISS.

    Ожидает конфиг с ``rag_pipeline.data.task='indexing'``.
    После индексации сохраняет FAISS-индекс и метаданные на диск
    в директорию ``cfg.paths.db_dir``.
    """
    cfg = setup_config(cfg)
    logger.info("Старт оффлайн-индексации базы знаний...")

    # 1. Загрузка токенизатора и энкодера
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    # 2. Инициализация векторной базы
    vector_db = hydra.utils.instantiate(cfg.vector_db)

    # 3. Подготовка данных в режиме indexing
    datamodule = DataModule(data_cfg=cfg.rag_pipeline.data, tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()

    indexer = KnowledgeBaseIndexer(
        model=base_model,
        pooler=pooler,
        vector_db=vector_db,
        device=cfg.rag_pipeline.indexing.device,
        precision=cfg.rag_pipeline.indexing.precision,
        push_batch_size=cfg.rag_pipeline.indexing.push_batch_size,
    )

    indexer.index_dataloader(dataloader, text_column="text")

    # 5. Сохранение индекса и метаданных на диск
    db_dir = Path(cfg.paths.db_dir)
    vector_db.save(db_dir)
    logger.info("Индекс сохранён в '%s'. База знаний готова к поиску.", db_dir)


if __name__ == "__main__":
    index_database()
