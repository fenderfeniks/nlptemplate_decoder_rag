# Makefile
# ==========================================
# КОМАНДЫ ДЛЯ РАЗРАБОТКИ (Task Runner)
# ==========================================
.PHONY: help install api train airflow down clean

# Команда по умолчанию
help:
	@echo "Доступные команды:"
	@echo "  make install   - Установить зависимости локально через uv"
	@echo "  make api       - Запустить продакшен сервер API"
	@echo "  make train     - Запустить задачу дообучения модели (Finetuning)"
	@echo "  make airflow   - Поднять локальный оркестратор для тестов DAG-ов"
	@echo "  make down      - Остановить все Docker контейнеры"
	@echo "  make clean     - Очистить кэш, логи и временные файлы"

# --- Локальная настройка ---
install:
	uv venv
	uv pip install -e ".[dev,training,rag,api]"

# --- Запуск Docker контейнеров ---
api:
	@echo "🚀 Запуск API..."
	# ИСПРАВЛЕНИЕ: Удален вызов несуществующего сервиса qdrant
	docker compose up -d --build api

train:
	@echo "🧠 Запуск изолированного обучения (Trainer)..."
	docker compose run --rm trainer python -m src.train

airflow:
	@echo "⏳ Поднятие Airflow..."
	docker compose up -d airflow

down:
	@echo "🛑 Остановка всех сервисов..."
	docker compose down

# --- Утилиты ---
clean:
	@echo "🧹 Очистка временных файлов и кэша..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .ruff_cache/