import inspect
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder
from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("Инициализация RAG-ретривера...")

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

    # Вытаскиваем поля для тестового прогона ДО instantiate —
    # они не являются параметрами RAGInferenceEmbedder.__init__()
    # и Hydra бросит TypeError если передать их через конфиг.
    inference_cfg = cfg.rag_pipeline.inference
    query: str = inference_cfg.get("test_query", "Тестовый запрос")
    top_k: int = inference_cfg.get("top_k", 3)

    # Фильтруем конфиг: оставляем только _target_ + параметры __init__ эмбеддера.
    # Читаем сигнатуру динамически — не хардкодим список ключей, чтобы не
    # рассинхронизироваться при добавлении новых параметров в RAGInferenceEmbedder.
    _valid_keys = frozenset(inspect.signature(RAGInferenceEmbedder.__init__).parameters) | {
        "_target_"
    }
    embedder_cfg = OmegaConf.masked_copy(
        inference_cfg, [k for k in inference_cfg if k in _valid_keys]
    )

    embedder = hydra.utils.instantiate(
        embedder_cfg,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # 3. Динамическая сборка Векторной БД
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)
    logger.info("Векторная БД загружена из '%s' (%d документов).", db_dir, vector_db.ntotal)

    # 4. Сборка ретривера
    retriever = hydra.utils.instantiate(
        cfg.rag_pipeline.retrieval,
        embedder=embedder,
        vector_db=vector_db,
    )

    logger.info("Запрос: '%s'", query)
    results = retriever.search(query, top_k=top_k)

    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        text = res.get("metadata", {}).get("text", "").replace("\n", " ")
        logger.info("[%d] score=%.4f | текст: %s...", i, score, text[:150])


if __name__ == "__main__":
    expected_pipeline = "rag_pipeline"

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
        sys.argv.append(f"pipeline_name={expected_pipeline}")

    infer()
