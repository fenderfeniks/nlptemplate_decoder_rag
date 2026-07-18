import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class RequestTimeLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        if "generate" in request.url.path or "chat" in request.url.path:
            logger.info(
                f"[{request.method}] {request.url.path} "
                f"| Статус: {response.status_code} "
                f"| Время генерации: {process_time:.2f} сек."
            )

        response.headers["X-Model-Process-Time"] = str(process_time)
        return response


# Изменили сигнатуру: теперь принимаем cors_origins
def setup_middlewares(app: FastAPI, cors_origins: list[str]):
    """
    Единая функция для регистрации всех middleware.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,  # Берем из конфига!
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestTimeLoggingMiddleware)
