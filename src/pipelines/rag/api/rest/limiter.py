# src/pipelines/rag/api/rest/limiter.py
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_real_ip(request: Request) -> str:
    """Надёжное извлечение реального IP-адреса клиента за Ingress/Proxy.

    Берёт первый IP из ``X-Forwarded-For`` — это истинный клиент.
    Последующие IP в списке — промежуточные прокси.

    Args:
        request: Входящий HTTP-запрос.

    Returns:
        IP-адрес клиента в виде строки.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


_env = os.getenv("ENVIRONMENT", "").lower()

# Лимитер отключается в тестовых окружениях или при явном флаге.
# _disabled используется как единственный источник правды для enabled= —
# ранее enabled= имел собственную логику которая не совпадала с _disabled.
_disabled: bool = (
    _env in {"testing", "test", "ci"} or os.getenv("DISABLE_RATE_LIMIT", "").lower() == "true"
)

limiter = Limiter(
    key_func=get_real_ip,
    default_limits=["20/minute"],
    enabled=not _disabled,
)
