# src/pipelines/rag/api/metrics.py
"""Метрики Prometheus для RAG Retrieval API.

Структура метрик:
    Инфраструктурные (HTTP-уровень):
        RAG_SEARCH_REQUESTS_TOTAL     — все входящие запросы (Counter)
        RAG_REQUEST_DURATION_SECONDS  — полное время от recv до last byte (Histogram)
        RAG_ERRORS_TOTAL              — ошибки по типу (Counter)

    Компонентные (pipeline-уровень):
        RAG_ENCODE_DURATION_SECONDS   — время энкодера (query -> vector)
        RAG_SEARCH_DURATION_SECONDS   — время поиска в Qdrant (без энкодера)
        RAG_RERANK_DURATION_SECONDS   — время реранкера (если включён)

    Качество результатов:
        RAG_RESULTS_RETURNED          — сколько документов вернул поиск (Histogram)
        RAG_TOP_SCORE                 — score первого результата (Histogram)
        RAG_EMPTY_RESULTS_TOTAL       — запросы с нулевым результатом (Counter)

    Состояние индекса:
        RAG_INDEX_TOTAL_DOCS          — ntotal в векторном хранилище (Gauge)

Лейблы:
    source      — "rest" (зарезервировано под будущий gRPC)
    error_type  — "encode_error" | "search_error" | "rerank_error" | "unknown"

Почему компонентные метрики разделены:
    RAG_REQUEST_DURATION включает сеть + все компоненты — по нему нельзя
    понять где именно деградация. Отдельные encode/search/rerank позволяют
    видеть в Grafana: «поиск в Qdrant вырос с 20ms до 200ms» без раскопок в логах.

Почему RAG_TOP_SCORE как Histogram, а не Gauge:
    Gauge показывает только последнее значение — бесполезно для p50/p95 анализа.
    Histogram позволяет строить percentile score по времени и видеть деградацию
    качества поиска (падение медианного score -> дрейф эмбеддингов или данных).
"""

from prometheus_client import Counter, Gauge, Histogram


# ---------------------------------------------------------------------------
# Инфраструктурные метрики
# ---------------------------------------------------------------------------

RAG_SEARCH_REQUESTS_TOTAL: Counter = Counter(
    "rag_search_requests_total",
    "Total number of RAG search requests received",
    ["source"],
)

RAG_REQUEST_DURATION_SECONDS: Histogram = Histogram(
    "rag_request_duration_seconds",
    "Total wall-clock time of a search request (recv -> response sent). "
    "Includes encoding, vector search, optional reranking, and serialization.",
    ["source", "status"],  # status: "success" | "error"
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

RAG_ERRORS_TOTAL: Counter = Counter(
    "rag_errors_total",
    "Total number of RAG pipeline errors by type",
    ["source", "error_type"],
    # error_type: "encode_error" | "search_error" | "rerank_error" | "unknown"
)


# ---------------------------------------------------------------------------
# Компонентные метрики (pipeline-уровень)
# ---------------------------------------------------------------------------

RAG_ENCODE_DURATION_SECONDS: Histogram = Histogram(
    "rag_encode_duration_seconds",
    "Time spent encoding the query into a vector (encoder inference only). "
    "Spike here -> GPU contention, model overload, or batch size too large.",
    ["source"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

RAG_SEARCH_DURATION_SECONDS: Histogram = Histogram(
    "rag_search_duration_seconds",
    "Time spent querying the vector store (Qdrant hybrid search: dense + BM25 + RRF). "
    "Excludes encoding. Spike here -> Qdrant overload, index size, or network latency.",
    ["source"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

RAG_RERANK_DURATION_SECONDS: Histogram = Histogram(
    "rag_rerank_duration_seconds",
    "Time spent reranking results with CrossEncoderReranker. "
    "Only recorded when reranker is active in the pipeline.",
    ["source"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


# ---------------------------------------------------------------------------
# Метрики качества результатов
# ---------------------------------------------------------------------------

RAG_RESULTS_RETURNED: Histogram = Histogram(
    "rag_results_returned",
    "Number of documents returned per search request. "
    "Consistently below top_k -> filters too strict or index too sparse.",
    ["source"],
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)

RAG_TOP_SCORE: Histogram = Histogram(
    "rag_top_score",
    "Cosine similarity score of the top-ranked document per request. "
    "Use p50/p95 to track retrieval quality drift over time. "
    "Falling median -> embedding drift, data distribution shift, or index corruption.",
    ["source"],
    # Диапазон [-1, 1] для косинусного сходства; плотность бакетов выше в [0.5, 1.0]
    # где живут качественные результаты — там нужна точность для алертинга
    buckets=(-1.0, 0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)

RAG_EMPTY_RESULTS_TOTAL: Counter = Counter(
    "rag_empty_results_total",
    "Number of search requests that returned zero documents. "
    "Spike -> score_threshold too high, index empty, or query out-of-distribution.",
    ["source"],
)


# ---------------------------------------------------------------------------
# Состояние индекса
# ---------------------------------------------------------------------------

RAG_INDEX_TOTAL_DOCS: Gauge = Gauge(
    "rag_index_total_docs",
    "Total number of documents currently indexed in the vector store. "
    "Updated on startup and after each indexing job. "
    "Drop to zero -> index corruption or failed mount.",
)