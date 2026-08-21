# src/api_gateway/metrics.py
from prometheus_client import Counter, Gauge, Histogram


# --- Gateway уровень ---
GATEWAY_REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Общее количество запросов к API Gateway",
    ["endpoint", "method", "status_code"],
)

GATEWAY_PROCESS_TIME = Histogram(
    "gateway_process_seconds",
    "Полное время обработки запроса (End-to-End latency)",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# --- LLM уровень ---
LLM_GENERATION_SECONDS = Histogram(
    "llm_generation_seconds",
    "Время генерации LLM (от отправки промпта до получения полного ответа)",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

LLM_TTFT_SECONDS = Histogram(
    "llm_ttft_seconds",
    "Time-to-first-token для стриминговых запросов",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

LLM_PROMPT_TOKENS_TOTAL = Counter(
    "llm_prompt_tokens_total",
    "Суммарное количество токенов в промптах",
    ["model"],
)

LLM_COMPLETION_TOKENS_TOTAL = Counter(
    "llm_completion_tokens_total",
    "Суммарное количество токенов в ответах",
    ["model"],
)

LLM_TOKENS_PER_SECOND = Histogram(
    "llm_tokens_per_second",
    "Скорость генерации токенов",
    ["model"],
    buckets=(1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0),
)

LLM_REQUESTS_IN_FLIGHT = Gauge(
    "llm_requests_in_flight",
    "Количество запросов к LLM в обработке прямо сейчас",
)

LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Количество ошибок при генерации",
    ["error_type"],
)

LLM_EMPTY_RESPONSES_TOTAL = Counter(
    "llm_empty_responses_total",
    "Количество пустых ответов от модели",
    ["model"],
)

LLM_TRUNCATED_RESPONSES_TOTAL = Counter(
    "llm_truncated_responses_total",
    "Количество ответов обрезанных по max_tokens",
    ["model"],
)