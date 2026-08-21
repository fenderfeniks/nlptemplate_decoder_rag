import os
import hashlib

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_real_ip(request: Request) -> str:
    """Надёжное извлечение реального IP-адреса клиента за Ingress/Proxy.

    Берёт первый IP из ``X-Forwarded-For`` — это истинный клиент.
    Последующие IP в списке — промежуточные прокси.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


def get_client_identifier(request: Request) -> str:
    """Извлекает идентификатор клиента для Rate Limiting.

    Если передан API-ключ в заголовке X-API-Key, используется его хэш
    (чтобы клиенты за одним NAT имели раздельные лимиты).
    Если ключа нет (например, на healthcheck или открытых эндпоинтах),
    используется реальный IP-адрес.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # Хэшируем ключ чтобы не держать сырой секрет в памяти лимитера
        return f"apikey:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
    
    return f"ip:{get_real_ip(request)}"


_env = os.getenv("ENVIRONMENT", "").lower()

# Лимитер отключается в тестовых окружениях или при явном флаге.
_disabled: bool = (
    _env in {"testing", "test", "ci"} or os.getenv("DISABLE_RATE_LIMIT", "").lower() == "true"
)

limiter = Limiter(
    key_func=get_client_identifier,  # <-- Изменили функцию извлечения ключа
    enabled=not _disabled,
)