# src/rag_pipeline/api/rest/limiter.py
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_real_ip(request: Request) -> str:
    """Извлечение реального IP-адреса клиента за Ingress/Proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=get_real_ip,
    default_limits=["20/minute"],  # Для RAG лимиты обычно мягче, чем для LLM
    enabled=os.getenv("ENVIRONMENT") != "testing",
)
