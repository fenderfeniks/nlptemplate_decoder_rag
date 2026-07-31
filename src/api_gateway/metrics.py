# src/api_gateway/metrics.py
from prometheus_client import Counter, Histogram


GATEWAY_REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Общее количество запросов к API Gateway",
    ["endpoint", "method", "status_code"],
)

GATEWAY_PROCESS_TIME = Histogram(
    "gateway_process_seconds",
    "Полное время обработки запроса (End-to-End latency)",
    ["endpoint"],
    # Бакеты: от 10 мс до 30 с — типичный диапазон для RAG+LLM
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
