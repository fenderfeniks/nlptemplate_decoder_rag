import logging
import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from src.api_gateway.metrics import GATEWAY_PROCESS_TIME, GATEWAY_REQUESTS_TOTAL


logger = logging.getLogger(__name__)


class GatewayTimeLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        status_code = 0

        async def send_with_logging(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                process_time = time.perf_counter() - start_time

                headers = dict(message.get("headers", []))
                headers[b"x-gateway-process-time"] = f"{process_time:.4f}".encode()
                message = {**message, "headers": list(headers.items())}

                endpoint = scope.get("path", "")
                GATEWAY_REQUESTS_TOTAL.labels(
                    endpoint=endpoint,
                    method=scope.get("method", ""),
                    status_code=str(status_code),
                ).inc()
                GATEWAY_PROCESS_TIME.labels(endpoint=endpoint).observe(process_time)

                logger.info(
                    "[%s] %s | Статус: %s | E2E Время: %.3f сек.",
                    scope.get("method", ""),
                    endpoint,
                    status_code,
                    process_time,
                )

            await send(message)

        await self.app(scope, receive, send_with_logging)


def setup_gateway_middlewares(app: FastAPI, cors_origins: list[str]) -> None:
    # Логирование внутри, CORS снаружи
    app.add_middleware(GatewayTimeLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
