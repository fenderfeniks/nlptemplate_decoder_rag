# src/api/rest/middlewares.py
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class RequestTimeLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware для логирования времени выполнения всех запросов."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Замеряет время запроса и добавляет заголовок X-Model-Process-Time.

        Args:
            request: Входящий HTTP-запрос.
            call_next: Следующий обработчик в цепочке.

        Returns:
            Ответ с заголовком X-Model-Process-Time.
        """
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time

        logger.info(
            "[%s] %s | Статус: %d | Время: %.3f сек.",
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )

        response.headers["X-Model-Process-Time"] = f"{process_time:.4f}"
        return response


def setup_middlewares(app: FastAPI, cors_origins: list[str]) -> None:
    """Регистрирует все Middleware в приложении FastAPI.

    Args:
        app: Экземпляр приложения FastAPI.
        cors_origins: Список разрешённых источников для CORS.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestTimeLoggingMiddleware)
