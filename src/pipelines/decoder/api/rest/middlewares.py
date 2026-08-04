# src/pipelines/decoder/api/rest/middlewares.py
import logging
import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """Чистый ASGI middleware для логирования времени запросов.

    Намеренно НЕ наследует ``BaseHTTPMiddleware``: тот буферизует весь
    response body в памяти при стриминге или исключениях, что критично
    для LLM-генерации где основной путь — ``StreamingResponse``.

    Добавляет заголовок ``X-Process-Time`` к каждому ответу.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        status_code: int = 0

        async def send_with_logging(message: dict[str, Any]) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                process_time = time.perf_counter() - start_time

                headers = dict(message.get("headers", []))
                headers[b"x-process-time"] = f"{process_time:.4f}".encode()
                message = {**message, "headers": list(headers.items())}

            await send(message)

        try:
            await self.app(scope, receive, send_with_logging)
        finally:
            process_time = time.perf_counter() - start_time
            logger.info(
                "[%s] %s | %d | %.3fs",
                scope.get("method", "?"),
                scope.get("path", "?"),
                status_code,
                process_time,
            )


def setup_middlewares(app: FastAPI, cors_origins: list[str]) -> None:
    """Регистрирует все middleware в приложении FastAPI.

    Порядок важен: Starlette применяет middleware в обратном порядке добавления.
    CORS должен быть внешним (добавлен последним) чтобы preflight-запросы
    обрабатывались до логирования.

    Args:
        app: Экземпляр приложения FastAPI.
        cors_origins: Список разрешённых источников для CORS.
    """
    # Логирование — внутренний слой
    app.add_middleware(RequestLoggingMiddleware)

    # CORS — внешний слой
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
