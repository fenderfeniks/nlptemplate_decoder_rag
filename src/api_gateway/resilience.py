# src/api_gateway/resilience.py
"""Circuit breaker для LLM и RAG downstream-сервисов.

Использует pybreaker. Установить: pip install pybreaker

Состояния автомата:
    CLOSED   → нормальная работа, запросы проходят.
    OPEN     → breaker сработал, запросы fast-fail'ятся немедленно
               (CircuitBreakerError без ожидания таймаута).
    HALF-OPEN → после recovery_timeout один пробный запрос:
               если успешен → CLOSED, иначе → OPEN снова.

Пороги настраиваются через env-переменные чтобы можно было
менять без передеплоя (через ConfigMap в k8s или .env).
"""

import logging
import os

import pybreaker

logger = logging.getLogger(__name__)


def _make_breaker(name: str, *, fail_env: str, timeout_env: str) -> pybreaker.CircuitBreaker:
    """Фабрика breaker'ов с чтением порогов из env."""
    fail_threshold = int(os.getenv(fail_env, "5"))
    recovery_timeout = int(os.getenv(timeout_env, "30"))

    breaker = pybreaker.CircuitBreaker(
        fail_max=fail_threshold,
        reset_timeout=recovery_timeout,
        name=name,
    )

    # Логируем смену состояния — это важное событие для on-call
    breaker.add_listeners(BreakerStateListener(name))
    return breaker


class BreakerStateListener(pybreaker.CircuitBreakerListener):
    """Логирует переходы состояний circuit breaker'а."""

    def __init__(self, name: str) -> None:
        self._name = name

    def state_change(self, cb, old_state, new_state) -> None:  # noqa: ANN001
        logger.warning(
            "CircuitBreaker [%s]: %s → %s (fail_count=%d)",
            self._name,
            old_state.name,
            new_state.name,
            cb.fail_counter,
        )


# --- Экземпляры breaker'ов ---
# Один breaker на downstream-сервис. Не создавай отдельные для каждого endpoint —
# если llama-server упал, все его endpoint'ы недоступны одновременно.

llm_breaker = _make_breaker(
    "llm-server",
    fail_env="LLM_BREAKER_FAIL_MAX",       # default: 5 ошибок подряд
    timeout_env="LLM_BREAKER_RESET_TIMEOUT",  # default: 30 сек
)

rag_breaker = _make_breaker(
    "rag-api",
    fail_env="RAG_BREAKER_FAIL_MAX",
    timeout_env="RAG_BREAKER_RESET_TIMEOUT",
)

# Публичный тип для isinstance-проверок в caller'ах
CircuitBreakerError = pybreaker.CircuitBreakerError