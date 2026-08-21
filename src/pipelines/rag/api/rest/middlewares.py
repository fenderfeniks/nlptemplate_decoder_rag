# src/pipelines/rag/api/rest/middlewares.py
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
    response body в памяти при стриминге или исключениях, что создаёт
    риск OOM при больших ответах и ломает стриминговые эндпоинты.

    Добавляет заголовок ``X-Process-Time`` к каждому ответу.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # Пропускаем WebSocket и lifespan события без замера
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        status_code: int = 0

        async def send_with_logging(message: dict[str, Any]) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                process_time = time.perf_counter() - start_time

                # Добавляем заголовок к уже существующим — не перезаписываем
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
    обрабатывались до логирования и rate limiting.

    Args:
        app: Экземпляр приложения FastAPI.
        cors_origins: Список разрешённых источников для CORS.
    """
    # Логирование — внутренний слой (добавлен первым -> выполняется последним снаружи)
    app.add_middleware(RequestLoggingMiddleware)

    # CORS — внешний слой (добавлен последним -> выполняется первым)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
