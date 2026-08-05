import inspect
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
from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder
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
    """
    cfg = setup_config(cfg)

    # Флаг incremental=false в схеме — дефолт означает "первый запуск".
    # Но если БД уже существует в манифесте, глупо требовать явного флага
    # при каждом обновлении корпуса. Поэтому:
    #   - incremental=true  → инкрементальный (явная просьба пользователя)
    #   - incremental=false → смотрим в манифест: есть vector_db_uri → авто-инкрементальный,
    #                         нет → полная переиндексация
    # Принудительная полная переиндексация: передать incremental=false
    # когда БД уже есть — это не поддерживается авто-логикой намеренно,
    # для этого нужно вручную удалить vector_db_uri из манифеста.
    user_requested_incremental: bool = cfg.get("incremental", False)

    # 1. Инициализация хранилища и роутера
    storage_client = hydra.utils.instantiate(cfg.storage)
    router = hydra.utils.instantiate(cfg.storage_router)
    uri_prefix = cfg.storage.uri_prefix
    manifest_uri = cfg.manifest.uri

    # 2. Единый резолвинг манифеста — читаем один раз, используем для энкодера и БД.
    #    resolve_and_patch патчит cfg.rag_pipeline.model.builder.model_name_or_path
    #    и возвращает (db_dir, lora_path).
    cache_base = Path(cfg.paths.model_dir) / "rag_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    try:
        db_dir, lora_path = resolver.resolve_and_patch(
            cfg, manifest_uri, pipeline_name="rag_pipeline"
        )
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов энкодера: %s", e)
        sys.exit(1)

    # Авто-определение режима: если пользователь не просил инкрементальный явно,
    # но БД уже есть в манифесте (db_dir не None) — переключаемся автоматически.
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

    # 3. Сборка энкодера
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

    # test_query/top_k живут в embedder.yaml для infer.py, но не являются
    # параметрами __init__ — фильтруем конфиг через сигнатуру чтобы не падать.
    _valid_keys = frozenset(inspect.signature(RAGInferenceEmbedder.__init__).parameters) | {
        "_target_"
    }
    embedder_cfg = OmegaConf.masked_copy(
        cfg.rag_pipeline.inference,
        [k for k in cfg.rag_pipeline.inference if k in _valid_keys],
    )
    embedder = hydra.utils.instantiate(
        embedder_cfg,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # 4. Инициализация векторной БД в зависимости от режима
    if incremental:
        # Инкрементальный режим — загружаем существующую БД чтобы переиспользовать doc_ids
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
        # Полная переиндексация — новая пустая БД.
        # Фильтруем loader из конфига: Hydra не должен его инстанцировать,
        # т.к. directory ещё не существует.
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
    dataloader = datamodule.train_dataloader()

    indexer = KnowledgeBaseIndexer(
        embedder=embedder,
        store=vector_db,
        push_batch_size=cfg.rag_pipeline.indexing.push_batch_size,
    )
    indexer.index_dataloader(dataloader, text_column="text")

    # 6. Сохранение локально и выгрузка в Storage
    local_db_dir = Path(cfg.paths.db_dir)
    vector_db.save(local_db_dir)
    logger.info("Индекс локально сохранён в '%s'.", local_db_dir)

    remote_db_dir = f"vector_dbs/{cfg.pipeline_name}_latest"
    logger.info("Выгрузка Векторной БД в Storage: %s", remote_db_dir)
    storage_client.upload(local_dir=local_db_dir, remote_path=remote_db_dir)

    # 7. Безопасное обновление манифеста.
    #    Скачиваем актуальный манифест чтобы не затереть ключи энкодера
    #    (model_uri / lora_uri / load_type). Обновляем только vector_db_uri.
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

    # Патчим cfg — сигнализируем что БД теперь существует.
    # Полезно если что-то downstream в том же процессе читает этот флаг.
    OmegaConf.update(cfg, "incremental", True, merge=True)

    logger.info(
        "Индексация завершена. Манифест обновлён. vector_db_uri: %s%s",
        uri_prefix,
        remote_db_dir,
    )


if __name__ == "__main__":
    expected_pipeline = "rag_pipeline"

    pipeline_arg_idx = next(
        (i for i, arg in enumerate(sys.argv) if arg.startswith("pipeline_name=")), None
    )

    if pipeline_arg_idx is not None:
        current_pipeline = sys.argv[pipeline_arg_idx].split("=")[1]
        if current_pipeline != expected_pipeline:
            logger.warning(
                "ВНИМАНИЕ! Передано pipeline_name=%s. Принудительно меняем на '%s'.",
                current_pipeline,
                expected_pipeline,
            )
            sys.argv[pipeline_arg_idx] = f"pipeline_name={expected_pipeline}"
    else:
        sys.argv.append(f"pipeline_name={expected_pipeline}")

    override_data = f"{expected_pipeline}/data=indexing"
    if not any(arg.startswith(f"{expected_pipeline}/data=") for arg in sys.argv):
        logger.info("Устанавливаем %s для корректной индексации.", override_data)
        sys.argv.append(override_data)

    index_database()
