import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class GatewayTimeLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            "[%s] %s | Статус: %d | E2E Время: %.2f сек.",
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )

        response.headers["X-Gateway-Process-Time"] = str(process_time)
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
