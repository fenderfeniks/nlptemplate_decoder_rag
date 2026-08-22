# scripts/rag_pipeline/index_db.py
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()

import hydra

from omegaconf import DictConfig, OmegaConf

from src.endpoints.index import run_universal_index
from src.pipelines.base.core.data.builder import DataModule
from src.pipelines.rag.inference.indexer import KnowledgeBaseIndexer
from src.pipelines.rag.inference.builder import build_inference_encoder
from src.tools.storage.resolver import ArtifactResolver
from src.utils.cli import enforce_pipeline
from src.utils.hydra_utils import setup_config


logger = logging.getLogger(__name__)


def run_index_logic(cfg: DictConfig, resolver: ArtifactResolver, router: Any) -> None:
    user_requested_incremental: bool = cfg.get("incremental", False)

    storage_client = hydra.utils.instantiate(cfg.system.storage)
    uri_prefix = cfg.system.storage.uri_prefix
    manifest_uri = cfg.system.manifest.uri
    pipeline_name = cfg.pipeline_name

    # 1. Загрузка артефактов (включая старую БД, если есть)
    db_dir, lora_path, *_ = resolver.resolve_and_patch(
        cfg, manifest_uri, pipeline_name="rag_pipeline", is_training=False
    )

    if user_requested_incremental:
        incremental = True
    elif db_dir is not None:
        logger.info("vector_db_uri найден в манифесте — инкрементальный режим.")
        incremental = True
    else:
        incremental = False

    # 2. Сборка энкодера
    base_model, pooler, tokenizer = build_inference_encoder(cfg, lora_path)
    embedder = hydra.utils.instantiate(
        cfg.inference.embedder,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # 3. Инициализация Векторной БД
    if incremental:
        if not db_dir:
            logger.critical("БД не найдена для инкрементального режима.")
            sys.exit(1)
        vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)
    else:
        vector_db_cfg = OmegaConf.create({
            k: v for k, v in OmegaConf.to_container(cfg.vector_db, resolve=True).items() if k != "loader"
        })
        vector_db = hydra.utils.instantiate(vector_db_cfg)

    # 4. Подготовка данных для индексации
    datamodule = DataModule(
        data_cfg=cfg.data,
        processed_data_dir=cfg.system.paths.processed_data_dir,
        tokenizer=tokenizer,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # 5. Процесс индексации
    lsh = hydra.utils.instantiate(cfg.inference.lsh) if cfg.inference.get("lsh") else None

    indexer = KnowledgeBaseIndexer(
        embedder=embedder,
        store=vector_db,
        lsh=lsh,  # <- добавить
        push_batch_size=cfg.inference.indexing.push_batch_size,
    )
    indexer.index_dataloader(datamodule.train_dataloader(), text_column="text")

    # 6. Сохранение БД и получение URI для манифеста
    #
    # Два сценария:
    #   - Qdrant: данные уже на сервере, save() — no-op, файлы не создаются.
    #     vector_db.get_uri() возвращает "qdrant://<url>/<collection>".
    #     Ничего не загружаем в storage — пишем server URI прямо в манифест.
    #
    #   - FAISS: save() пишет файлы на диск, get_uri() возвращает None.
    #     Загружаем файлы в storage и пишем storage URI в манифест.
    #
    db_uri: str | None = None

    # Проверяем, поддерживает ли бэкенд get_uri() (Qdrant и любые будущие серверные БД)
    if hasattr(vector_db, "get_uri"):
        server_uri = vector_db.get_uri()
    else:
        server_uri = None

    if server_uri is not None:
        # Qdrant — данные уже персистентны на сервере
        db_uri = server_uri
        logger.info("Vector DB персистентна на сервере: %s", db_uri)
    else:
        # FAISS — сохраняем файлы и загружаем в storage
        remote_db_dir = f"{pipeline_name}/vector_db"

        with tempfile.TemporaryDirectory() as tmp_db_dir:
            local_db_path = Path(tmp_db_dir) / "vector_db"
            vector_db.save(local_db_path)

            if not local_db_path.exists():
                logger.error(
                    "vector_db.save() не создал директорию '%s'. "
                    "Проверьте реализацию save() для данного бэкенда.",
                    local_db_path,
                )
                sys.exit(1)

            storage_client.upload(local_dir=local_db_path, remote_path=remote_db_dir)
            logger.info("Vector DB загружена в storage: %s", remote_db_dir)

        db_uri = f"{uri_prefix}{remote_db_dir}"

    # 7. Обновление манифеста

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            manifest = router.download_manifest(manifest_uri, cache_dir=tmp_path / "old_manifest")
        except Exception:
            manifest = {}

        if pipeline_name not in manifest:
            manifest[pipeline_name] = {}

        manifest[pipeline_name].update({
            "vector_db_uri": db_uri,
            "db_updated_at": datetime.now(timezone.utc).isoformat(),
        })

        manifest_filename = manifest_uri.rstrip("/").split("/")[-1]
        manifest_file = tmp_path / manifest_filename
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        router.upload_file_to_uri(manifest_file, manifest_uri)

    OmegaConf.update(cfg, "incremental", True, merge=True)
    logger.info("Индексация завершена. vector_db_uri: %s", db_uri)


@hydra.main(config_path="../../configs", config_name="index_rag", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    run_universal_index(cfg, "rag_pipeline", run_index_logic)


if __name__ == "__main__":
    enforce_pipeline("rag_pipeline")
    main()