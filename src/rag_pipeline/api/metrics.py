# src/rag_pipeline/api/metrics.py
"""Определение метрик Prometheus для RAG API."""

from prometheus_client import Counter, Histogram


# Счетчик общего числа запросов к поиску
RAG_SEARCH_REQUESTS_TOTAL: Counter = Counter(
    "rag_search_requests_total",
    "Total number of RAG search requests",
    ["source"],
)

# Гистограмма времени инференса энкодера + поиска в FAISS
RAG_SEARCH_TIME: Histogram = Histogram(
    "rag_search_seconds", "Time spent encoding query and searching in Vector DB", ["source"]
)

# Опционально: можно добавить Gauge для отслеживания количества документов в базе,
# если база поддерживает динамическое обновление.
