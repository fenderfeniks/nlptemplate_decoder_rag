# src/api_gateway/middlewares.py
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api_gateway.metrics import GATEWAY_PROCESS_TIME, GATEWAY_REQUESTS_TOTAL


logger = logging.getLogger(__name__)


class GatewayTimeLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time

        endpoint = request.url.path
        status_code = str(response.status_code)

        logger.info(
            "[%s] %s | Статус: %s | E2E Время: %.3f сек.",
            request.method,
            endpoint,
            status_code,
            process_time,
        )

        # Prometheus-метрики
        GATEWAY_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=request.method,
            status_code=status_code,
        ).inc()
        GATEWAY_PROCESS_TIME.labels(endpoint=endpoint).observe(process_time)

        response.headers["X-Gateway-Process-Time"] = f"{process_time:.4f}"
        return response


def setup_gateway_middlewares(app: FastAPI, cors_origins: list[str]) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GatewayTimeLoggingMiddleware)
