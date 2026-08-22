# scripts/rag/eval.py
"""Оценка качества RAG-ретривера на замороженном эталонном бенчмарке.

Логика сборки компонентов идентична infer.py (единый источник правды).
Инфраструктурные блоки (logger, router, resolver, benchmark, export, drift)
вынесены в src/endpoints/eval.py и переиспользуются с decoder/eval.py.

Двухэтапная оценка:
    bi-encoder recall  — не потерял ли энкодер релевантные документы.
    cross-encoder NDCG — качество финальной выдачи после реранкинга.
Разделение позволяет точно локализовать деградацию в пайплайне.
"""

import logging

from dotenv import load_dotenv


load_dotenv()

import hydra  # noqa
from omegaconf import DictConfig, OmegaConf  # noqa

from src.endpoints.eval import EvalContext, run_universal_eval  # noqa
from src.evaluation.evaluators.retriever import RetrieverEvaluator  # noqa
from src.pipelines.rag.inference.builder import build_inference_encoder  # noqa
from src.utils.cli import enforce_pipeline  # noqa
from src.utils.hydra_utils import setup_config  # noqa
from src.utils.logger import setup_logging  # noqa


setup_logging()
logger = logging.getLogger(__name__)


def _build_reranker(cfg: DictConfig, ctx: EvalContext):
    """Собирает реранкер по той же логике, что и infer.py.

    Использует cfg.use_reranker (bool) — единый флаг для обоих скриптов.
    При ошибке логирует и возвращает None (деградация без краша).
    """
    if not cfg.get("use_reranker", False):
        return None

    try:
        _, reranker_lora, _ = ctx.resolver.resolve_and_patch(
            cfg,
            cfg.system.manifest.uri,
            pipeline_name="reranker_pipeline",
            is_training=False,
        )
        # Переопределяем auto_model_class для классификатора (как в infer.py)
        OmegaConf.update(
            cfg,
            "model.builder.auto_model_class",
            "transformers.AutoModelForSequenceClassification",
        )
        r_model, _, r_tokenizer = build_inference_encoder(cfg, reranker_lora)
        reranker = hydra.utils.instantiate(
            cfg.inference.reranker,
            model=r_model,
            tokenizer=r_tokenizer,
        )
        logger.info("Реранкер загружен.")
        return reranker
    except Exception as e:
        logger.error("Сбой загрузки реранкера — продолжаем без него: %s", e)
        return None


def _build_and_eval(ctx: EvalContext) -> dict[str, float]:
    """Pipeline-специфичная функция: сборка RAG + батчевая оценка.

    Вызывается из run_universal_eval после загрузки бенчмарка.
    Сигнатура фиксирована: (EvalContext) -> dict[str, float].
    """
    cfg = ctx.cfg
    inference_cfg = cfg.inference

    # ── 1. Энкодер (идентично infer.py) ──────────────────────────────────
    base_model, pooler, tokenizer = build_inference_encoder(cfg, ctx.lora_path)
    embedder = hydra.utils.instantiate(
        inference_cfg.embedder,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )

    # ── 2. Векторная БД (идентично infer.py) ─────────────────────────────
    # ctx.db_dir — Path (FAISS) или "qdrant://..." (Qdrant), проверен в run_universal_eval
    vector_db = hydra.utils.instantiate(cfg.vector_db.loader, directory=ctx.db_dir)
    logger.info("Векторная БД загружена. Документов: %d.", vector_db.ntotal)

    # ── 3. Реранкер (идентично infer.py, флаг use_reranker) ──────────────
    reranker = _build_reranker(cfg, ctx)

    # ── 4. Ретривер ───────────────────────────────────────────────────────
    retriever = hydra.utils.instantiate(
        inference_cfg.retriever,
        embedder=embedder,
        vector_db=vector_db,
        reranker=reranker,
    )

    # ── 5. Параметры оценки ───────────────────────────────────────────────
    retrieval_top_k: int = inference_cfg.get("top_k", 20)
    rerank_top_k: int = inference_cfg.get("rerank_top_k", 5)

    evaluator = RetrieverEvaluator(
        retriever=retriever,
        tokenizer=tokenizer,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        ood_threshold=inference_cfg.get("ood_threshold", 0.3),
        # drift_cfg намеренно не передаём: внутренний _check_drift в RetrieverEvaluator
        # делал sys.exit(1) ДО шага экспорта метрик в run_universal_eval.
        # Drift check выполняется централизованно в run_universal_eval после экспорта.
        drift_cfg={},
        raise_on_drift=False,
    )

    # ── 6. Батчевая оценка ────────────────────────────────────────────────
    with ctx.experiment_logger.start_run(run_name="rag_hybrid_eval"):
        metrics = evaluator.evaluate(
            queries=ctx.queries,
            ground_truths=ctx.ground_truths,
            metrics_logger=ctx.experiment_logger,
            stage="test",
        )

    # ── 7. Итоговый отчёт ─────────────────────────────────────────────────
    reranker_active = reranker is not None
    logger.info("=" * 60)
    logger.info("Итог оценки (%s реранкер):", "активен" if reranker_active else "отключён")
    logger.info(
        "  Bi-encoder recall@%d : %.4f",
        retrieval_top_k,
        metrics.get(f"recall_{retrieval_top_k}_biencoder", 0.0),
    )
    if reranker_active:
        logger.info(
            "  Cross-encoder ndcg@%d : %.4f",
            rerank_top_k,
            metrics.get(f"ndcg_{rerank_top_k}", 0.0),
        )
    else:
        logger.info(
            "  NDCG@%d (bi-encoder only): %.4f",
            rerank_top_k,
            metrics.get(f"ndcg_{rerank_top_k}", 0.0),
        )
    logger.info("  OOD rate              : %.4f", metrics.get("ood_rate", 0.0))
    logger.info("  Avg context tokens    : %.1f", metrics.get("avg_context_tokens", 0.0))
    logger.info("  Total latency ms/q    : %.1f", metrics.get("total_latency_ms_per_query", 0.0))
    logger.info("=" * 60)

    return metrics


@hydra.main(config_path="../../configs", config_name="eval_rag", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("Старт независимой оценки RAG (Retrieval Evaluation)...")

    data_cfg = cfg.get("data", {})

    run_universal_eval(
        cfg=cfg,
        pipeline_name="rag_pipeline",
        build_and_eval_fn=_build_and_eval,
        query_column=data_cfg.get("query_column", "question"),
        doc_id_column=data_cfg.get("ground_truth_column", "chunk_id"),
        cache_subdir="rag_cache",
        require_db=True,
        # drift_metric_key по умолчанию берётся из cfg в run_universal_eval;
        # для RAG типичный дефолт — recall_<k>_biencoder, его нужно прописать в конфиге:
        # drift_metric_key: recall_20_biencoder
        # drift_threshold: 0.8
    )


if __name__ == "__main__":
    enforce_pipeline("rag_pipeline")
    evaluate()
