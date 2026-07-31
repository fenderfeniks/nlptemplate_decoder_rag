# src/api/rest/limiter.py
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_real_ip(request: Request) -> str:
    """Надёжное извлечение реального IP-адреса клиента за Ingress/Proxy.

    Args:
        request: Входящий HTTP-запрос.

    Returns:
        IP-адрес клиента в виде строки.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Берём первый IP из списка (истинный клиент)
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


# Лимитер отключается при ENVIRONMENT=testing|test|ci (регистронезависимо)
# или при явном DISABLE_RATE_LIMIT=true
_env = os.getenv("ENVIRONMENT", "").lower()
_disabled = (
    _env in {"testing", "test", "ci"} or os.getenv("DISABLE_RATE_LIMIT", "").lower() == "true"
)

limiter = Limiter(
    key_func=get_real_ip,
    default_limits=["10/minute"],
    enabled=not _disabled,
)
