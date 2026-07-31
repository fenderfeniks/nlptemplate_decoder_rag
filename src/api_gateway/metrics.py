from prometheus_client import Counter, Histogram


GATEWAY_REQUESTS_TOTAL = Counter(
    "gateway_requests_total", "Общее количество запросов к API Gateway", ["endpoint"]
)

GATEWAY_PROCESS_TIME = Histogram(
    "gateway_process_seconds", "Полное время обработки запроса (End-to-End latency)", ["endpoint"]
)
