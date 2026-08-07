# scripts/rag/index_db.py
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines.base.core.data.builder import DataModule
from src.pipelines.rag.indexing.indexer import KnowledgeBaseIndexer
from src.pipelines.rag.inference.builder import build_inference_encoder
from src.pipelines.rag.inference.embedder_factory import build_embedder
from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def index_database(cfg: DictConfig) -> None:
    """Оффлайн-индексация базы знаний.

    Режимы (передаётся через CLI):
        incremental=false (дефолт) — полная переиндексация: создаём новую пустую БД.
            Используется после смены модели или полного обновления корпуса.
        incremental=true — инкрементальное обновление: загружаем существующую БД,
            добавляем только новые документы (дубли пропускаются через doc_id).

    Авто-логика: если БД уже есть в манифесте, при incremental=false (дефолт)
    переключаемся в инкрементальный режим автоматически. Для принудительной
    полной переиндексации — удалите vector_db_uri из манифеста вручную.
    """
    cfg = setup_config(cfg)
    user_requested_incremental: bool = cfg.get("incremental", False)

    # 1. Инициализация хранилища и роутера
    storage_client = hydra.utils.instantiate(cfg.storage)
    router = hydra.utils.instantiate(cfg.storage_router)
    uri_prefix = cfg.storage.uri_prefix
    manifest_uri = cfg.manifest.uri

    # 2. Резолвинг манифеста: патчит model_name_or_path, возвращает (db_dir, lora_path)
    cache_base = Path(cfg.paths.model_dir) / "rag_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    try:
        db_dir, lora_path = resolver.resolve_and_patch(
            cfg, manifest_uri, pipeline_name="rag_pipeline"
        )
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов энкодера: %s", e)
        sys.exit(1)

    # Авто-определение режима
    if user_requested_incremental:
        incremental = True
    elif db_dir is not None:
        logger.info(
            "vector_db_uri найден в манифесте — автоматически переключаемся "
            "в инкрементальный режим. Для полной переиндексации удалите "
            "vector_db_uri из манифеста вручную."
        )
        incremental = True
    else:
        incremental = False

    logger.info(
        "Старт индексации базы знаний. Режим: %s.",
        "инкрементальный" if incremental else "полная переиндексация",
    )

    # 3. Сборка энкодера и эмбеддера
    base_model, pooler, tokenizer = build_inference_encoder(cfg, lora_path)
    embedder = build_embedder(cfg, base_model, pooler, tokenizer)

    # 4. Инициализация векторной БД
    if incremental:
        if not db_dir:
            logger.critical(
                "Инкрементальный режим запрошен, но манифест не содержит 'vector_db_uri'. "
                "Запустите без incremental=true для первичной индексации."
            )
            sys.exit(1)
        logger.info("Загрузка существующей БД из '%s'...", db_dir)
        vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)
        logger.info("БД загружена. Документов в индексе: %d.", vector_db.ntotal)
    else:
        # Фильтруем loader: directory ещё не существует, Hydra не должен его трогать
        logger.info("Создание новой пустой БД...")
        vector_db_cfg = OmegaConf.create(
            {
                k: v
                for k, v in OmegaConf.to_container(cfg.vector_db, resolve=True).items()
                if k != "loader"
            }
        )
        vector_db = hydra.utils.instantiate(vector_db_cfg)

    # 5. Индексация корпуса
    datamodule = DataModule(data_cfg=cfg.rag_pipeline.data, tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    indexer = KnowledgeBaseIndexer(
        embedder=embedder,
        store=vector_db,
        push_batch_size=cfg.rag_pipeline.indexing.push_batch_size,
    )
    indexer.index_dataloader(datamodule.train_dataloader(), text_column="text")

    # 6. Сохранение локально и выгрузка в Storage
    local_db_dir = Path(cfg.paths.db_dir)
    vector_db.save(local_db_dir)
    logger.info("Индекс локально сохранён в '%s'.", local_db_dir)

    remote_db_dir = f"vector_dbs/{cfg.pipeline_name}_latest"
    logger.info("Выгрузка Векторной БД в Storage: %s", remote_db_dir)
    storage_client.upload(local_dir=local_db_dir, remote_path=remote_db_dir)

    # 7. Безопасное обновление манифеста (только vector_db_uri — не трогаем model_uri/lora_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            manifest = router.download_manifest(manifest_uri, cache_dir=tmp_path / "old_manifest")
            logger.info("Найден существующий манифест. Обновляем vector_db_uri.")
        except Exception:
            logger.warning("Существующий манифест не найден. Будет создан новый.")
            manifest = {}

        manifest["vector_db_uri"] = f"{uri_prefix}{remote_db_dir}"
        manifest["db_updated_at"] = datetime.now(timezone.utc).isoformat()

        manifest_file = tmp_path / f"{cfg.pipeline_name}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload(local_dir=tmp_dir, remote_path="manifests")

    OmegaConf.update(cfg, "incremental", True, merge=True)
    logger.info(
        "Индексация завершена. Манифест обновлён. vector_db_uri: %s%s",
        uri_prefix,
        remote_db_dir,
    )


if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline

    enforce_pipeline("rag_pipeline", "rag_pipeline/data=indexing")
    index_database()
