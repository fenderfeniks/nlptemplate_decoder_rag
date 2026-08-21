# src/api_gateway/telemetry.py
"""OpenTelemetry + Jaeger: настройка distributed tracing.

Использование в lifespan (server.py):
    from src.api_gateway.telemetry import setup_telemetry
    setup_telemetry(app, service_name="api-gateway")

Зависимости (добавить в requirements):
    opentelemetry-sdk
    opentelemetry-exporter-otlp-proto-grpc
    opentelemetry-instrumentation-fastapi
    opentelemetry-instrumentation-httpx
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def setup_telemetry(app, service_name: str = "api-gateway") -> None:
    """Инициализирует OpenTelemetry трейсинг и автоинструментацию.

    Экспортирует спаны в Jaeger через OTLP/gRPC.
    Адрес Jaeger читается из OTEL_EXPORTER_OTLP_ENDPOINT (default: http://jaeger:4317).

    Автоинструментация покрывает:
        - Все входящие FastAPI-роуты (заголовки traceparent пробрасываются автоматически).
        - Все исходящие httpx-запросы (к RAG API) — trace-context передаётся в заголовках.

    Ручные спаны для LLM и внутренней логики добавляются через get_tracer(__name__)
    непосредственно в точках вызова (chat.py, orchestrator.py, inference.py).
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")

    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("APP_VERSION", "unknown"),
        "deployment.environment": os.getenv("APP_ENV", "production"),
    })

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Автоматически добавляет middleware для входящих HTTP-запросов.
    # Читает/пишет заголовки traceparent/tracestate (W3C Trace Context).
    FastAPIInstrumentor.instrument_app(app)

    # Пробрасывает trace-контекст во все исходящие httpx-запросы.
    # Это позволяет RAG API видеть тот же trace_id и строить сквозные трейсы.
    HTTPXClientInstrumentor().instrument()

    logger.info(
        "OpenTelemetry инициализирован (service=%s, exporter=%s)",
        service_name, endpoint,
    )