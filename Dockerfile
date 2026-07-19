# ==========================================
# STAGE 1: BUILDER (Сборка зависимостей)
# ==========================================
FROM python:3.10-slim AS builder

# Устанавливаем системные зависимости для компиляции C++ расширений
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /build

COPY pyproject.toml README.md ./
ARG INSTALL_EXTRAS="api,rag"

# Создаем изолированное виртуальное окружение
RUN uv venv /opt/venv

# ИСПРАВЛЕНИЕ: Устанавливаем пакеты в .venv с принудительным скачиванием GPU-колес PyTorch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    ".[${INSTALL_EXTRAS}]"

# ==========================================
# STAGE 2: RUNNER (Чистый боевой образ)
# ==========================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PROJECT_ROOT=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Копируем ТОЛЬКО готовое виртуальное окружение из билдера (без build-essential и кэшей)
COPY --from=builder /opt/venv /opt/venv

RUN addgroup --system mlgroup && adduser --system --group mluser

COPY src/ ./src/
COPY configs/ ./configs/

RUN chown -R mluser:mlgroup /app
USER mluser

CMD ["python", "-m", "src.run_api"]