FROM python:3.10-slim

# 1. Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Копируем бинарник uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Настройки среды
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PROJECT_ROOT=/app

WORKDIR /app

# 4. Создаем non-root пользователя (Требование безопасности K8s)
RUN addgroup --system mlgroup && adduser --system --group mluser

# 5. КЭШИРОВАНИЕ ЗАВИСИМОСТЕЙ (Магия Docker Layers)
# Копируем ТОЛЬКО файлы конфигурации.
# Пока pyproject.toml не изменится, этот слой будет мгновенно браться из кэша Docker.
COPY pyproject.toml README.md ./

ARG INSTALL_EXTRAS="api,rag"

# Просим uv скомпилировать и установить зависимости, но не устанавливать сам исходный код проекта.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[${INSTALL_EXTRAS}]"

# 6. КОПИРУЕМ ИСХОДНИКИ
# Копируем код и конфиги только сейчас. Это изолирует изменения кода от установки библиотек.
COPY src/ ./src/
COPY configs/ ./configs/

# 7. Финальная сборка и выдача прав
# Передаем права нашему безопасному пользователю
RUN chown -R mluser:mlgroup /app

# Переключаемся на безопасного пользователя
USER mluser

# 8. Точка входа по умолчанию
CMD ["python", "-m", "src.run_api"]