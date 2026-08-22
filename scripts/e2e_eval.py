# scripts/e2e_eval.py
"""End-to-End оценка системы через API Gateway.

Что делает:
    1. Читает benchmark.jsonl из S3 (через StorageRouter + манифест).
    2. Отправляет запросы к /api/v1/chat/stream Gateway.
    3. Считает ROUGE-1 между ответами Gateway и эталонными ответами.
    4. Проверяет пороги — при нарушении завершается с sys.exit(1).

Env-переменные:
    GATEWAY_URL         — URL Gateway (default: http://localhost:8000)
    PIPELINE_NAME       — секция манифеста (default: rag_pipeline)
    ROUGE_THRESHOLD     — минимальный ROUGE-1 (default: 0.2)
    LATENCY_P95_MAX_S   — максимальный p95 latency в секундах (default: 30.0)
    MAX_SAMPLES         — сколько записей брать из бенчмарка (default: 50)
    REQUEST_TIMEOUT_S   — таймаут одного запроса к Gateway (default: 60.0)

Запуск:
    python -m scripts.e2e_eval
    GATEWAY_URL=http://nlp-gateway:8000 python -m scripts.e2e_eval
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import httpx
import hydra
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.utils.logger import setup_logging


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация через env
# ---------------------------------------------------------------------------

GATEWAY_URL: str = os.getenv("GATEWAY_URL", "http://localhost:8000")
PIPELINE_NAME: str = os.getenv("PIPELINE_NAME", "rag_pipeline")
ROUGE_THRESHOLD: float = float(os.getenv("ROUGE_THRESHOLD", "0.2"))
LATENCY_P95_MAX_S: float = float(os.getenv("LATENCY_P95_MAX_S", "30.0"))
MAX_SAMPLES: int = int(os.getenv("MAX_SAMPLES", "50"))
REQUEST_TIMEOUT_S: float = float(os.getenv("REQUEST_TIMEOUT_S", "60.0"))

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_DIR = str(_SCRIPT_DIR.parent / "configs")


# ---------------------------------------------------------------------------
# ROUGE-1 (без внешних зависимостей)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Простая токенизация: lowercase + split по пробелам и пунктуации."""
    import re

    return re.findall(r"\b\w+\b", text.lower())


def rouge1_f1(hypothesis: str, reference: str) -> float:
    """ROUGE-1 F1 между гипотезой и референсом.

    Реализован вручную чтобы не тащить rouge-score как зависимость.
    Для production-оценки замени на rouge_score.compute().
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    hyp_counts: dict[str, int] = {}
    for t in hyp_tokens:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1

    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = sum(min(hyp_counts.get(t, 0), ref_counts[t]) for t in ref_counts)

    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Gateway клиент
# ---------------------------------------------------------------------------


def _call_gateway(client: httpx.Client, query: str) -> tuple[str, float]:
    """Отправляет запрос к Gateway, собирает стриминговый ответ.

    Returns:
        (response_text, latency_s)

    Raises:
        httpx.HTTPError: при сетевых ошибках или 4xx/5xx.
    """
    start = time.perf_counter()
    chunks: list[str] = []

    with client.stream(
        "POST",
        f"{GATEWAY_URL}/api/v1/chat/stream",
        json={"query": query},
        timeout=REQUEST_TIMEOUT_S,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if chunk:
                chunks.append(chunk)

    latency = time.perf_counter() - start
    return "".join(chunks), latency


def _health_check(client: httpx.Client) -> None:
    """Проверяет доступность Gateway перед прогоном."""
    try:
        resp = client.get(f"{GATEWAY_URL}/health", timeout=10.0)
        resp.raise_for_status()
        logger.info("Gateway health check OK: %s", GATEWAY_URL)
    except Exception as e:
        logger.error(
            "Gateway недоступен (%s). Убедитесь что деплоймент запущен и GATEWAY_URL корректен.",
            e,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Загрузка бенчмарка
# ---------------------------------------------------------------------------


def _load_benchmark(router, manifest_uri: str) -> list[dict]:
    """Скачивает benchmark.jsonl из S3 через StorageRouter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifest = router.download_manifest(manifest_uri, cache_dir=tmp_path / "manifest")

        if PIPELINE_NAME not in manifest:
            logger.error(
                "Секция '%s' не найдена в манифесте. Доступные: %s",
                PIPELINE_NAME,
                list(manifest.keys()),
            )
            sys.exit(1)

        benchmark_uri = manifest[PIPELINE_NAME].get("benchmark_uri")
        if not benchmark_uri:
            logger.error(
                "benchmark_uri не найден в манифесте для '%s'. Сначала запустите build_benchmark.",
                PIPELINE_NAME,
            )
            sys.exit(1)

        local_path = tmp_path / "benchmark.jsonl"
        router.download_file_from_uri(benchmark_uri, local_path)

        records = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    logger.info("Бенчмарк загружен: %d записей (pipeline=%s)", len(records), PIPELINE_NAME)
    return records


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------


def _run_eval(router, manifest_uri: str) -> None:
    records = _load_benchmark(router, manifest_uri)

    if not records:
        logger.error("Бенчмарк пуст — нечего оценивать.")
        sys.exit(1)

    samples = records[:MAX_SAMPLES]
    logger.info("Прогон e2e eval: %d запросов → Gateway %s", len(samples), GATEWAY_URL)

    rouge_scores: list[float] = []
    latencies: list[float] = []
    errors = 0

    with httpx.Client() as client:
        _health_check(client)

        for i, record in enumerate(samples, 1):
            # RAG-бенчмарк: поле question + answer
            # LLM-бенчмарк: поле prompt + response
            query = record.get("question") or record.get("prompt", "")
            reference = record.get("answer") or record.get("response", "")

            if not query or not reference:
                logger.warning("[%d/%d] Пропускаем запись без query/reference", i, len(samples))
                continue

            try:
                response_text, latency = _call_gateway(client, query)
                score = rouge1_f1(response_text, reference)

                rouge_scores.append(score)
                latencies.append(latency)

                logger.info(
                    "[%d/%d] ROUGE-1=%.3f latency=%.2fs | query='%s...'",
                    i,
                    len(samples),
                    score,
                    latency,
                    query[:60],
                )

            except Exception as e:
                errors += 1
                logger.warning("[%d/%d] Ошибка запроса: %s", i, len(samples), e)

    # ---------------------------------------------------------------------------
    # Агрегация и проверка порогов
    # ---------------------------------------------------------------------------

    if not rouge_scores:
        logger.error("Ни один запрос не выполнен успешно.")
        sys.exit(1)

    avg_rouge = statistics.mean(rouge_scores)
    min_rouge = min(rouge_scores)
    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95_latency = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]
    error_rate = errors / len(samples)

    logger.info("=" * 60)
    logger.info("E2E Eval Results (pipeline=%s)", PIPELINE_NAME)
    logger.info("  Запросов:       %d / %d успешно", len(rouge_scores), len(samples))
    logger.info("  Error rate:     %.1f%%", error_rate * 100)
    logger.info("  ROUGE-1 avg:    %.4f  (порог: %.4f)", avg_rouge, ROUGE_THRESHOLD)
    logger.info("  ROUGE-1 min:    %.4f", min_rouge)
    logger.info("  Latency p95:    %.2fs (порог: %.2fs)", p95_latency, LATENCY_P95_MAX_S)
    logger.info("  Latency avg:    %.2fs", statistics.mean(latencies))
    logger.info("=" * 60)

    failed = False

    if avg_rouge < ROUGE_THRESHOLD:
        logger.error(
            "ROUGE-1 avg %.4f < порога %.4f — качество ответов недостаточное.",
            avg_rouge,
            ROUGE_THRESHOLD,
        )
        failed = True

    if p95_latency > LATENCY_P95_MAX_S:
        logger.error(
            "Latency p95 %.2fs > порога %.2fs — система слишком медленная.",
            p95_latency,
            LATENCY_P95_MAX_S,
        )
        failed = True

    if error_rate > 0.1:
        logger.error(
            "Error rate %.1f%% > 10%% — слишком много упавших запросов.",
            error_rate * 100,
        )
        failed = True

    if failed:
        sys.exit(1)

    logger.info("E2E eval пройден успешно.")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=== E2E Eval: старт (pipeline=%s) ===", PIPELINE_NAME)

    config_dir = os.getenv("HYDRA_CONFIG_DIR", _DEFAULT_CONFIG_DIR)
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="eval_rag")
        OmegaConf.resolve(cfg)

    router = hydra.utils.instantiate(cfg.system.storage_router)
    manifest_uri: str = cfg.system.manifest.uri

    _run_eval(router, manifest_uri)


if __name__ == "__main__":
    main()
