# src/api_gateway/rag_logger.py
"""Структурированное логирование RAG-троек (query, retrieved_docs, response).

Почему отдельный модуль, а не просто logger.info:
    - Тройки — это observability data, а не application logs.
      Их нужно отправлять в отдельный sink (файл / индекс ES).
    - Payload большой (тексты документов). Смешивать с app-логами
      засоряет stdout и усложняет парсинг в Logstash/Fluentd.
    - Отдельный логгер позволяет настроить ротацию и retention независимо.

Настройка в logging.yaml / dictConfig:
    loggers:
      rag.traces:
        level: INFO
        handlers: [rag_file_handler]
        propagate: false   # не дублировать в root-логгер

    handlers:
      rag_file_handler:
        class: logging.handlers.RotatingFileHandler
        filename: logs/rag_traces.jsonl
        maxBytes: 104857600   # 100MB
        backupCount: 7
        formatter: json_formatter

    formatters:
      json_formatter:
        (): pythonjsonlogger.jsonlogger.JsonFormatter
        format: "%(asctime)s %(name)s %(levelname)s %(message)s"

Если хочешь писать сразу в Elasticsearch минуя файл — замени handler
на cmreshandler (pip install CMRESHandler) или используй Filebeat sidecar.

В span'е OpenTelemetry пишем только doc_ids и scores (не полный текст),
чтобы не раздувать Jaeger. Корреляция по request_id позволяет связать
тройку в Kibana с трейсом в Jaeger.
"""

import json
import logging
import time
from typing import Any


_rag_logger = logging.getLogger("rag.traces")


def log_rag_triple(
    *,
    request_id: str,
    query: str,
    retrieved_docs: list[dict[str, Any]],
    response: str,
    model: str,
    rag_degraded: bool,
    elapsed_s: float,
    trace_id: str | None = None,
) -> None:
    """Записывает тройку (query, docs, response) в структурированный лог.

    Args:
        request_id:     UUID запроса (для корреляции с app-логами и трейсами).
        query:          Оригинальный запрос пользователя.
        retrieved_docs: Список документов от RAG API.
                        Каждый элемент: {"id": ..., "score": ..., "text": ...}
        response:       Итоговый ответ LLM (очищенный _batch_cleaner'ом).
        model:          Имя модели.
        rag_degraded:   True если RAG был недоступен и ответ без контекста.
        elapsed_s:      E2E latency в секундах.
        trace_id:       OTEL trace_id (hex str) для cross-ссылки с Jaeger.
                        Если None — OTEL не настроен или трейс не активен.
    """
    # Для хранения в JSONL пишем только нужные поля из документов.
    # Полный текст — в поле text, но можно урезать если места мало.
    doc_summaries = [
        {
            "id": doc.get("id") or doc.get("metadata", {}).get("id"),
            "score": doc.get("score"),
            "text_preview": (doc.get("metadata", {}).get("text", "") or "")[:200],
        }
        for doc in retrieved_docs
    ]

    record = {
        "event": "rag_triple",
        "ts": time.time(),
        "request_id": request_id,
        "trace_id": trace_id,
        "query": query,
        "doc_count": len(retrieved_docs),
        "docs": doc_summaries,
        "response_length": len(response),
        "response_preview": response[:500],
        "model": model,
        "rag_degraded": rag_degraded,
        "elapsed_s": round(elapsed_s, 3),
    }

    # message — пустая строка, вся информация в структурированных полях.
    # JsonFormatter pythonjsonlogger сериализует extra-поля автоматически.
    # Если JsonFormatter не настроен — fallback на json.dumps в message.
    _rag_logger.info(json.dumps(record, ensure_ascii=False))
