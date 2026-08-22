# src/api_gateway/metrics_recorder.py
import asyncio
import logging

import httpx

from src.api_gateway.metrics import (
    LLM_COMPLETION_TOKENS_TOTAL,
    LLM_EMPTY_RESPONSES_TOTAL,
    LLM_ERRORS_TOTAL,
    LLM_GENERATION_SECONDS,
    LLM_PROMPT_TOKENS_TOTAL,
    LLM_REQUESTS_IN_FLIGHT,
    LLM_TOKENS_PER_SECOND,
    LLM_TTFT_SECONDS,
)


logger = logging.getLogger(__name__)


def record_ttft(*, model: str, ttft_s: float) -> None:
    LLM_TTFT_SECONDS.labels(model=model).observe(ttft_s)


def record_error(*, error_type: str) -> None:
    LLM_ERRORS_TOTAL.labels(error_type=error_type).inc()


def classify_error(exc: Exception) -> str:
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return "timeout"
    return "llm_error"


def record_stream_metrics(
    *,
    request_id: str,
    model: str,
    status: str,
    elapsed: float,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_s: float,
    generated_text: str,
) -> None:
    LLM_REQUESTS_IN_FLIGHT.dec()
    LLM_GENERATION_SECONDS.labels(model=model).observe(elapsed)

    if prompt_tokens:
        LLM_PROMPT_TOKENS_TOTAL.labels(model=model).inc(prompt_tokens)
    if completion_tokens:
        LLM_COMPLETION_TOKENS_TOTAL.labels(model=model).inc(completion_tokens)
        if elapsed > 0:
            LLM_TOKENS_PER_SECOND.labels(model=model).observe(completion_tokens / elapsed)

    if not generated_text.strip():
        LLM_EMPTY_RESPONSES_TOTAL.labels(model=model).inc()
        logger.warning("[%s] пустой ответ от модели", request_id)

    logger.info(
        "[%s] stream завершён | model=%s status=%s ttft=%.2fs elapsed=%.2fs",
        request_id,
        model,
        status,
        ttft_s,
        elapsed,
    )
