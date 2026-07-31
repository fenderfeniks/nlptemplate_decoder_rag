# Makefile
# ==========================================
# КОМАНДЫ ДЛЯ РАЗРАБОТКИ (Task Runner)
# ==========================================
.PHONY: help install train_decoder train_rag api_gateway api_rag mlflow clean docker_train_decoder docker_train_rag docker_api docker_airflow docker_down

help:
	@echo "Доступные команды:"
	@echo "--- Локальная разработка ---"
	@echo "  make install             - Установить зависимости локально через uv"
	@echo "  make train_decoder       - Запустить обучение LLM локально"
	@echo "  make train_rag           - Запустить обучение RAG энкодера локально"
	@echo "  make api_gateway         - Запустить локальный API Gateway (Оркестратор)"
	@echo "  make api_rag             - Запустить локальный RAG API"
	@echo "  make mlflow              - Запустить локальный сервер MLflow UI"
	@echo "  make clean               - Очистить кэш, логи и временные файлы"
	@echo ""
	@echo "--- Docker окружение ---"
	@echo "  make docker_train_decoder - Обучение LLM в изолированном контейнере"
	@echo "  make docker_train_rag     - Обучение RAG в изолированном контейнере"
	@echo "  make docker_api           - Поднять всю инфраструктуру API (Gateway, RAG, LLM, Demo)"
	@echo "  make docker_airflow       - Поднять локальный оркестратор (Airflow) в Docker"
	@echo "  make docker_down          - Остановить все Docker контейнеры"

# --- Локальные команды ---

install:
	uv venv
	uv pip install -e ".[dev,training,api]"

train_decoder:
	@echo "🧠 Запуск локального обучения LLM..."
	python -m scripts.decoder_pipeline.train $(ARGS)

train_rag:
	@echo "🧠 Запуск локального обучения RAG..."
	python -m scripts.rag_pipeline.train $(ARGS)

api_gateway:
	@echo "🚀 Запуск локального API Gateway (Оркестратор)..."
	python -m src.api_gateway.run_api

api_rag:
	@echo "🚀 Запуск локального RAG API (Энкодер + Поиск)..."
	python -m src.rag_pipeline.api.run_api

mlflow:
	@echo "📊 Запуск MLflow UI..."
	mlflow ui --backend-store-uri sqlite:///logs/mlflow.db --default-artifact-root ./logs/mlartifacts --host 127.0.0.1 --port 5000

clean:
	@echo "🧹 Очистка временных файлов и кэша..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .ruff_cache/

# --- Docker команды ---

docker_train_decoder:
	@echo "🐳 Запуск изолированного обучения LLM (Docker)..."
	docker compose run --rm train_decoder $(ARGS)

docker_train_rag:
	@echo "🐳 Запуск изолированного обучения RAG (Docker)..."
	docker compose run --rm train_rag $(ARGS)

docker_api:
	@echo "🐳 Запуск API инфраструктуры в Docker..."
	docker compose up -d --build api_gateway rag_api llm_api demo

docker_airflow:
	@echo "⏳ Поднятие Airflow в Docker..."
	docker compose up -d airflow

docker_down:
	@echo "🛑 Остановка всех Docker сервисов..."
	docker compose down