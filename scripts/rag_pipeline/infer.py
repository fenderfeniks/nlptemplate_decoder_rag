import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.endpoints.infer import run_universal_infer
from src.pipelines.rag.inference.builder import build_inference_encoder
from src.tools.storage.resolver import ArtifactResolver
from src.utils.cli import enforce_pipeline
from src.utils.hydra_utils import setup_config


load_dotenv()
logger = logging.getLogger(__name__)


def run_rag_logic(cfg: DictConfig, resolver: ArtifactResolver) -> None:
    """Специфичная логика сборки и инференса RAG."""
    # 1. Загрузка артефактов (БД и адаптер)
    db_dir, lora_path, *_ = resolver.resolve_and_patch(
        cfg, cfg.system.manifest.uri, pipeline_name="rag_pipeline", is_training=False
    )
    if not db_dir:
        raise ValueError("Манифест не содержит 'vector_db_uri'. База не найдена.")

    # 2. Сборка энкодера (RAG биэнкодер)
    base_model, pooler, tokenizer = build_inference_encoder(cfg, lora_path)
    embedder = hydra.utils.instantiate(
        cfg.inference.embedder,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # 3. Загрузка векторной БД
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)
    logger.info("Векторная БД загружена. Документов: %d.", vector_db.ntotal)

    # 4. Реранкер (опционально)
    reranker = None
    if cfg.get("use_reranker", False):
        try:
            _, reranker_lora, *_ = resolver.resolve_and_patch(
                cfg, cfg.system.manifest.uri, pipeline_name="reranker_pipeline", is_training=False
            )
            # Переопределяем auto_model_class для реранкера
            OmegaConf.update(
                cfg,
                "model.builder.auto_model_class",
                "transformers.AutoModelForSequenceClassification",
            )
            r_model, _, r_tokenizer = build_inference_encoder(cfg, reranker_lora)
            reranker = hydra.utils.instantiate(
                cfg.inference.reranker, model=r_model, tokenizer=r_tokenizer
            )
            logger.info("Реранкер загружен.")
        except Exception as e:
            logger.error("Сбой загрузки реранкера — продолжаем без него: %s", e)

    # 5. Сборка ретривера
    retriever = hydra.utils.instantiate(
        cfg.inference.retriever, embedder=embedder, vector_db=vector_db, reranker=reranker
    )

    # 6. Тестовый запрос
    inference_cfg = cfg.inference
    query: str = inference_cfg.get("test_query", "Тестовый запрос")
    top_k: int = inference_cfg.get("top_k", 5)

    logger.info("Запрос: '%s'", query)
    results = retriever.search(query=query, top_k=top_k)

    reranker_active = reranker is not None
    logger.info("Результаты (%s реранкер):", "активен" if reranker_active else "отключён")

    for i, res in enumerate(results, 1):
        dense_score = res.get("score", 0.0)
        ce_score = res.get("cross_encoder_score")
        text = res.get("metadata", {}).get("text", "").replace("\n", " ")

        if ce_score is not None:
            logger.info(
                "[%d] dense=%.4f | ce=%.4f | текст: %s...", i, dense_score, ce_score, text[:150]
            )
        else:
            logger.info("[%d] dense=%.4f | текст: %s...", i, dense_score, text[:150])


@hydra.main(config_path="../../configs", config_name="eval_rag", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    run_universal_infer(cfg, "rag_pipeline", run_rag_logic)


if __name__ == "__main__":
    enforce_pipeline("rag_pipeline")
    main()
