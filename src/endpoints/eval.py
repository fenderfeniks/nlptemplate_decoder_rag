# src/endpoints/eval.py
"""Общий эндпоинт для eval-скриптов (RAG и Decoder).

Инкапсулирует повторяющиеся блоки:
    - инициализация experiment_logger / router / resolver
    - резолвинг артефактов через ArtifactResolver
    - загрузка бенчмарка через BenchmarkLoader
    - экспорт метрик в JSON
    - drift check (универсальный, направление задаётся флагом)

Использование (RAG):
    from src.endpoints.eval import run_universal_eval, EvalContext

    def build_and_eval(ctx: EvalContext) -> dict[str, float]:
        retriever = ...
        evaluator = RetrieverEvaluator(...)
        with ctx.experiment_logger.start_run(run_name="rag_eval"):
            return evaluator.evaluate(...)

    run_universal_eval(cfg, "rag_pipeline", build_and_eval)

Использование (Decoder):
    def build_and_eval(ctx: EvalContext) -> dict[str, float]:
        model, tokenizer = build_decoder_model(cfg, ctx.lora_path)
        evaluator = DecoderEvaluator(...)
        with ctx.experiment_logger.start_run(run_name="decoder_eval"):
            return evaluator.evaluate(...)

    run_universal_eval(cfg, "decoder_pipeline", build_and_eval)
"""

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from src.tools.benchmark.loader import BenchmarkLoader
from src.tools.storage.resolver import ArtifactResolver


logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    """Собранный контекст оценки — передаётся в pipeline-специфичную функцию.

    Attributes:
        cfg:               Hydra-конфиг (полный).
        experiment_logger: Инициализированный логгер экспериментов.
        router:            StorageRouter для доступа к артефактам.
        resolver:          ArtifactResolver с уже настроенным cache_base.
        lora_path:         Path к LoRA-адаптеру (None если не найден).
        db_dir:            Path / URI к векторной БД (None для decoder).
        queries:           Список запросов из бенчмарка.
        ground_truths:     Список эталонных ответов / doc_id (зависит от пайплайна).
        benchmark_dataset: Исходный HuggingFace Dataset для дополнительного доступа.
        extra:             Словарь для pipeline-специфичных данных без расширения датакласса.
    """

    cfg: DictConfig
    experiment_logger: Any
    router: Any
    resolver: ArtifactResolver
    lora_path: Path | None
    db_dir: Path | str | None
    queries: list[str]
    ground_truths: list[Any]
    benchmark_dataset: Any  # HuggingFace Dataset
    extra: dict[str, Any] = field(default_factory=dict)


def _init_infrastructure(
    cfg: DictConfig,
    pipeline_name: str,
    cache_subdir: str,
) -> tuple[Any, Any, ArtifactResolver]:
    """Инициализирует experiment_logger, router и resolver."""
    experiment_logger = hydra.utils.instantiate(cfg.system.logger.experiment_logger)
    router = hydra.utils.instantiate(cfg.system.storage_router)
    cache_base = Path(cfg.system.paths.model_dir) / cache_subdir
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)
    return experiment_logger, router, resolver


def _resolve_artifacts(
    cfg: DictConfig,
    resolver: ArtifactResolver,
    pipeline_name: str,
) -> tuple[Path | str | None, Path | None]:
    """Резолвит артефакты из манифеста.

    Returns:
        (db_dir, lora_path) — db_dir=None для decoder-пайплайнов.
    """
    try:
        db_dir, lora_path, _ = resolver.resolve_and_patch(
            cfg,
            cfg.system.manifest.uri,
            pipeline_name=pipeline_name,
            is_training=False,
        )
        return db_dir, lora_path
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов [%s]: %s", pipeline_name, e)
        sys.exit(1)


def _resolve_column(
    dataset: Any,
    requested: str | None,
    fallbacks: list[str],
    role: str,
) -> str:
    """Возвращает реальное имя колонки в датасете.

    Сначала проверяет явно запрошенную колонку, затем перебирает fallbacks.
    При полном промахе логирует доступные колонки и делает sys.exit(1).

    Args:
        dataset:   HuggingFace Dataset с атрибутом column_names.
        requested: Имя из конфига / аргумента (может быть None).
        fallbacks: Список имён-кандидатов в порядке приоритета.
        role:      Человекочитаемая роль колонки для сообщения об ошибке.
    """
    available: list[str] = list(dataset.column_names)
    candidates = ([requested] if requested else []) + fallbacks
    for name in candidates:
        if name and name in available:
            if name != requested:
                logger.info(
                    "Колонка '%s' (%s) не найдена — используем '%s'.",
                    requested,
                    role,
                    name,
                )
            return name

    logger.error(
        "Не найдена колонка для '%s'. "
        "Запрошено: '%s'. Проверено: %s. Доступные колонки: %s. "
        "Укажите правильное имя в cfg.data или передайте явно в run_universal_eval.",
        role,
        requested,
        candidates,
        available,
    )
    sys.exit(1)


def _load_benchmark(
    cfg: DictConfig,
    router: Any,
    cache_base: Path,
    pipeline_name: str,
    query_column: str | None = None,
    answer_column: str | None = None,
    doc_id_column: str | None = None,
) -> tuple[Any, list[str], list[Any]]:
    """Загружает эталонный бенчмарк через BenchmarkLoader.

    Колонки резолвятся через _resolve_column: сначала явно переданное имя,
    затем fallback-список типичных имён. Это защищает от KeyError при
    расхождении между конфигом и реальной схемой бенчмарка.

    Returns:
        (dataset, queries, ground_truths)
        ground_truths — список строк (decoder) или список списков id (RAG).
    """
    benchmark_loader = BenchmarkLoader(
        router=router,
        cache_dir=cache_base / "benchmark",
        manifest_uri=cfg.system.manifest.uri,
        pipeline_name=pipeline_name,
    )

    # BenchmarkLoader.load_as_dataset принимает колонки для переименования/валидации.
    # Передаём только то, что задано явно — не хотим навязывать дефолты на уровне API.
    load_kwargs: dict[str, str] = {}
    if query_column:
        load_kwargs["query_column"] = query_column
    if answer_column:
        load_kwargs["answer_column"] = answer_column
    if doc_id_column:
        load_kwargs["doc_id_column"] = doc_id_column

    dataset = benchmark_loader.load_as_dataset(**load_kwargs)

    if dataset is None or len(dataset) == 0:
        logger.error(
            "Эталонный бенчмарк не найден или пустой [%s]. "
            "Убедитесь что benchmark_uri прописан в манифесте.",
            pipeline_name,
        )
        sys.exit(1)

    logger.info("Бенчмарк загружен: %d записей.", len(dataset))

    # Автодетект реальных имён колонок с внятной диагностикой при промахе
    real_query_col = _resolve_column(
        dataset,
        query_column,
        fallbacks=["question", "query", "input", "text", "prompt"],
        role="query_column",
    )
    real_gt_col: str | None = None
    if doc_id_column or answer_column:
        real_gt_col = _resolve_column(
            dataset,
            doc_id_column or answer_column,
            fallbacks=["chunk_id", "doc_id", "id", "response", "answer", "target", "output"],
            role="ground_truth_column",
        )

    queries = [item[real_query_col] for item in dataset]

    # RAG: ground_truths — списки doc_id; Decoder: строки-ответы
    ground_truths: list[Any]
    if real_gt_col:
        ground_truths = [
            (
                item[real_gt_col]
                if isinstance(item[real_gt_col], list)
                else [item[real_gt_col]]
                if doc_id_column  # для RAG всегда оборачиваем в список
                else item[real_gt_col]
            )
            for item in dataset
        ]
    else:
        ground_truths = [[] for _ in dataset]

    return dataset, queries, ground_truths


def check_drift(
    metrics: dict[str, Any],
    drift_threshold: float,
    metric_key: str = "test_rouge1",
    lower_is_better: bool = False,
) -> None:
    """Проверяет основную метрику на дрифт и делает sys.exit(1) при деградации.

    Args:
        metrics:          Словарь метрик после оценки.
        drift_threshold:  Порог деградации.
        metric_key:       Ключ основной метрики в словаре.
        lower_is_better:  True для loss-метрик (test_loss и подобных).
    """
    primary_metric = metrics.get(metric_key)
    if primary_metric is None:
        logger.warning("Ключ '%s' не найден в метриках для drift check.", metric_key)
        return

    logger.info(
        "Drift check: %s=%.4f, порог=%.4f (lower_is_better=%s)",
        metric_key,
        primary_metric,
        drift_threshold,
        lower_is_better,
    )

    degraded = (
        primary_metric > drift_threshold if lower_is_better else primary_metric < drift_threshold
    )
    if degraded:
        logger.error("ДРИФТ (деградация %s). Выход с кодом 1.", metric_key)
        sys.exit(1)


def export_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Сохраняет метрики в JSON-файл."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info("Метрики экспортированы в %s", output_path)


def run_universal_eval(
    cfg: DictConfig,
    pipeline_name: str,
    build_and_eval_fn: Callable[[EvalContext], dict[str, float]],
    *,
    query_column: str | None = None,
    answer_column: str | None = None,
    doc_id_column: str | None = None,
    cache_subdir: str | None = None,
    require_db: bool = False,
) -> dict[str, float]:
    """Универсальный оркестратор eval-скрипта.

    Собирает инфраструктуру, загружает бенчмарк, вызывает pipeline-специфичную
    функцию оценки, экспортирует метрики и выполняет drift check.

    Args:
        cfg:                Hydra-конфиг.
        pipeline_name:      Имя пайплайна ('rag_pipeline' / 'decoder_pipeline').
        build_and_eval_fn:  Функция (EvalContext) -> dict[str, float].
                            Должна инициализировать модель и запустить оценку.
        query_column:       Колонка запросов в бенчмарке (дефолт из cfg.data).
        answer_column:      Колонка ответов (для decoder).
        doc_id_column:      Колонка doc_id (для RAG).
        cache_subdir:       Поддиректория кеша (дефолт: '<pipeline_name>_cache').
        require_db:         Если True — падает при отсутствии db_dir в манифесте.

    Returns:
        Словарь метрик (пустой при ошибке загрузки бенчмарка).
    """
    _cache_subdir = cache_subdir or f"{pipeline_name}_cache"
    experiment_logger, router, resolver = _init_infrastructure(cfg, pipeline_name, _cache_subdir)
    cache_base = Path(cfg.system.paths.model_dir) / _cache_subdir

    db_dir, lora_path = _resolve_artifacts(cfg, resolver, pipeline_name)
    if require_db and not db_dir:
        logger.critical("Манифест не содержит 'vector_db_uri'. База не найдена.")
        sys.exit(1)

    # Резолвим колонки из cfg.data, если не переданы явно.
    # Дефолтов здесь нет намеренно — автодетект с fallback-цепочкой
    # выполняется в _load_benchmark._resolve_column после загрузки датасета,
    # когда известны реальные column_names.
    data_cfg = cfg.get("data", {})
    _query_col = query_column or data_cfg.get("query_column")
    _answer_col = answer_column or data_cfg.get("target_column") or data_cfg.get("answer_column")
    _doc_id_col = doc_id_column or data_cfg.get("ground_truth_column")

    dataset, queries, ground_truths = _load_benchmark(
        cfg=cfg,
        router=router,
        cache_base=cache_base,
        pipeline_name=pipeline_name,
        query_column=_query_col,
        answer_column=_answer_col,
        doc_id_column=_doc_id_col,
    )

    ctx = EvalContext(
        cfg=cfg,
        experiment_logger=experiment_logger,
        router=router,
        resolver=resolver,
        lora_path=lora_path,
        db_dir=db_dir,
        queries=queries,
        ground_truths=ground_truths,
        benchmark_dataset=dataset,
    )

    metrics = build_and_eval_fn(ctx)

    if not metrics:
        logger.warning("Оценка не вернула метрик.")
        return {}

    # Экспорт
    metrics_file = cfg.get("metrics_output_path", "metrics.json")
    export_metrics(metrics, metrics_file)

    # Drift check
    drift_threshold = cfg.get("drift_threshold")
    if drift_threshold is not None:
        drift_metric_key = cfg.get("drift_metric_key", "test_rouge1")
        is_lower_better = drift_metric_key in ("test_loss",)
        check_drift(
            metrics,
            drift_threshold=drift_threshold,
            metric_key=drift_metric_key,
            lower_is_better=is_lower_better,
        )

    return metrics
