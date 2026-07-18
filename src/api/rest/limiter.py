# src/api/rest/limiter.py
import os

from slowapi import Limiter
from slowapi.util import get_remote_address


# Создаем глобальный инстанс лимитера
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["10/minute"],
    enabled=os.getenv("ENVIRONMENT") != "testing",
)
