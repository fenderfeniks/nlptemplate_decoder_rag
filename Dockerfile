# Убираем AS builder, так как сборка одностадийная
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем бинарник uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Настройки Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Сначала копируем только файлы описания проекта
COPY pyproject.toml README.md ./

# Копируем исходники и конфиги
COPY src/ ./src/
COPY configs/ ./configs/

# Группы зависимостей по умолчанию
ARG INSTALL_EXTRAS="api,rag"

# Магия DevOps: используем кэш-монтирование Docker.
# Даже если код в src/ изменится, uv мгновенно достанет библиотеки из кэша, а не будет качать их из сети.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e ".[${INSTALL_EXTRAS}]"

# Правильный запуск через НАШ файл-оркестратор (который сам читает .env и порты)
CMD ["python", "-m", "src.run_api"]