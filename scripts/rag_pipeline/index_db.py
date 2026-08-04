import json
import logging
import tempfile
from datetime import datetime, timezone
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
    cfg = setup_config(cfg)
    logger.info("Старт оффлайн-индексации базы знаний...")

    # Инициализация хранилища и роутера (для чтения старого манифеста)
    storage_client = hydra.utils.instantiate(cfg.storage)
    router = hydra.utils.instantiate(cfg.storage_router)
    uri_prefix = cfg.storage.uri_prefix.rstrip("/")

    # 1. Загрузка токенизатора и энкодера
    tokenizer = hydra.utils.instantiate(cfg.rag_pipeline.model.tokenizer).build()
    builder = hydra.utils.instantiate(cfg.rag_pipeline.model.builder)
    base_model = builder.build(tokenizer=tokenizer)
    pooler = hydra.utils.instantiate(cfg.rag_pipeline.model.pooling)

    # 2. Инициализация векторной базы
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=None)

    # 3. Индексация
    datamodule = DataModule(data_cfg=cfg.rag_pipeline.data, tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()

    indexer = KnowledgeBaseIndexer(
        embedder=hydra.utils.instantiate(
            cfg.rag_pipeline.inference, model=base_model, pooler=pooler, tokenizer=tokenizer
        ),
        store=vector_db,
        push_batch_size=cfg.rag_pipeline.indexing.push_batch_size,
    )
    indexer.index_dataloader(dataloader, text_column="text")

    # 4. Сохранение локально
    local_db_dir = Path(cfg.paths.db_dir)
    vector_db.save(local_db_dir)
    logger.info("Индекс локально сохранён в '%s'.", local_db_dir)

    # 5. Выгрузка в Storage
    remote_db_dir = f"vector_dbs/{cfg.pipeline_name}_latest"
    logger.info("Выгрузка Векторной БД в Storage: %s", remote_db_dir)
    storage_client.upload(local_dir=local_db_dir, remote_path=remote_db_dir)

    # 6. Обновление манифеста
    manifest_uri = cfg.get(
        "manifest_uri", f"{uri_prefix}/manifests/{cfg.pipeline_name}_manifest.json"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            # Пытаемся скачать текущий манифест, чтобы не затереть веса энкодера
            manifest = router.download_manifest(manifest_uri, cache_dir=tmp_path / "old_manifest")
            logger.info("Найден существующий манифест. Обновляем векторную БД.")
        except Exception:
            logger.warning("Существующий манифест не найден. Будет создан новый.")
            manifest = {}

        manifest["vector_db_uri"] = f"{uri_prefix}/{remote_db_dir}"
        manifest["db_updated_at"] = datetime.now(timezone.utc).isoformat()

        manifest_file = tmp_path / f"{cfg.pipeline_name}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload(local_dir=tmp_dir, remote_path="manifests")

    logger.info("Индексация завершена. Манифест обновлен.")


if __name__ == "__main__":
    index_database()
