# Industrial NLP Template — Decoder + RAG

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.2%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-89b4fa?logo=python&logoColor=white)](https://hydra.cc/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Airflow](https://img.shields.io/badge/Airflow-2.8%2B-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-blue)](https://github.com/facebookresearch/faiss)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-orange)](https://github.com/huggingface/peft)
[![uv](https://img.shields.io/badge/uv-fast%20packaging-DE5FE9)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/linter-ruff-black)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready шаблон для задач **генерации текста (Decoder / LLM)** и **семантического поиска (RAG)**. Цель — от датасета до работающего API-эндпоинта за минимальное количество шагов, без написания инфраструктурного кода с нуля.

---

## Зачем этот шаблон

Большинство NLP-задач — seq2seq-перевод, RAG-чат-боты, суммаризация, ответы на вопросы — повторяют одну и ту же инфраструктуру: загрузка и предобработка данных, fine-tuning с LoRA, трекинг экспериментов, подача модели через REST API, переиндексация базы документов. Этот шаблон реализует всё это **один раз и продакшен-качественно**, оставляя вам только то, что специфично для вашей задачи: данные, промпты и гиперпараметры.

**Что получаете "из коробки":**

- Полный пайплайн дообучения LLM (SFT / CPT) с LoRA/QLoRA через PyTorch Lightning
- Полный пайплайн дообучения RAG-энкодера (contrastive learning, MNRL / Triplet loss)
- Векторная база FAISS с поддержкой Flat и HNSW-индексов
- Три REST-сервиса (API Gateway, RAG API, LLM API через vLLM) с мониторингом Prometheus + Grafana
- Airflow DAG-и для регулярного переобучения, переиндексации и продвижения модели
- Helm-чарт для деплоя в Kubernetes
- Streamlit-демо для быстрой проверки результата
- Jupyter-ноутбуки для EDA, prompt engineering и оценки качества
- Полный тест-сьют (pytest) с моками тяжёлых зависимостей

---

## Быстрый старт

### 1. Установка

```bash
# Клонируй репозиторий и перейди в папку
git clone <your-repo-url> && cd nlp-template

# Установи зависимости через uv (dev + training + api)
make install

# Создай .env из шаблона и заполни токены
cp .env.example .env
```

Минимально необходимые переменные в `.env`:

```dotenv
HF_TOKEN=your_hf_token_here          # для gated-моделей (Llama, Phi и т.д.)
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 2. Подготовь данные

Шаблон поддерживает загрузку из HuggingFace Hub (по умолчанию), локальных файлов и Kaggle. Укажи источник в конфиге:

```yaml
# configs/decoder_pipeline/data/source/hf.yaml
dataset_name: "your-org/your-dataset"
split: "train"
```

Или переключись на локальный CSV/JSON через `decoder_pipeline/data/source=local`.

### 3. Выбери модель

Готовые конфиги архитектур лежат в `configs/decoder_pipeline/model/architecture/`:

| Конфиг | Модель | Описание |
|---|---|---|
| `Qwen2.5-1.5B.yaml` | Qwen/Qwen2.5-1.5B | Компактная, работает на 8 GB VRAM |
| `Qwen3-4B-Instruct-2507.yaml` | Qwen/Qwen3-4B-Instruct-2507 | Актуальный instruct-вариант |
| `phi-4-mini-inst.yaml` | microsoft/Phi-4-mini-instruct | Microsoft SSM-модель |
| `local.yaml` | (путь на диске) | Любая локальная модель |

Для RAG-энкодера — `configs/rag_pipeline/model/architecture/`:

| Конфиг | Модель |
|---|---|
| `bge-m3.yaml` | BAAI/bge-m3 (мультиязычный) |
| `local.yaml` | Любой локальный энкодер |

### 4. Запусти обучение

```bash
# Дообучение LLM (SFT, LoRA, bf16)
make train_decoder

# Дообучение RAG-энкодера (contrastive learning)
make train_rag

# Переключить модель через CLI (без редактирования файлов)
make train_decoder ARGS="decoder_pipeline/model/architecture=Qwen3-4B-Instruct-2507"

# Включить 4-bit QLoRA для экономии VRAM
make train_decoder ARGS="decoder_pipeline/model/quantization=4bit"

# Увеличить количество шагов
make train_decoder ARGS="training.max_steps=5000"
```

### 5. Смотри результаты

```bash
# Запусти MLflow UI
make mlflow
# → http://127.0.0.1:5000
```

Каждый прогон автоматически логирует метрики (loss, BLEU/ROUGE для декодера, MRR/Recall@K для RAG), гиперпараметры и LoRA-адаптер как MLflow-артефакт.

### 6. Подними API

```bash
# Весь стек одной командой (Docker)
make docker_api

# Или локально без Docker
make api_gateway   # → http://localhost:8000
make api_rag       # → http://localhost:8001
```

LLM-инференс обслуживается через **vLLM** с OpenAI-совместимым API — переключение между моделями без изменения кода.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                      CLIENT / TG BOT                    │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│              API Gateway  (FastAPI, :8000)               │
│   RAGOrchestrator: запрос → RAG API → промпт → LLM API  │
│   PromptManager: шаблоны из YAML, без хардкода          │
│   Middleware: CORS, rate-limit, Prometheus-метрики       │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
┌────────────▼──────────┐  ┌──────────▼──────────────────┐
│  RAG API  (FastAPI,   │  │  LLM API  (vLLM, :8002)     │
│  :8001)               │  │  OpenAI-compatible /v1       │
│  RAGInferenceEmbedder │  │  Любая CausalLM-модель       │
│  FAISSVectorDB        │  │                              │
└───────────────────────┘  └──────────────────────────────┘

ОБУЧЕНИЕ (batch jobs):
  scripts/decoder_pipeline/train.py  — SFT/CPT + LoRA merge
  scripts/rag_pipeline/train.py      — Contrastive + индексация
  → трекинг: MLflow
  → оркестрация: Airflow DAG (KubernetesPodOperator)
  → деплой: Helm chart
```

---

## Структура проекта

```
.
├── configs/                     # Вся конфигурация (Hydra)
│   ├── main.yaml                # Точка входа, собирает пайплайн
│   ├── decoder_pipeline/        # Модель, данные, обучение, инференс
│   │   ├── model/architecture/  # Qwen, Phi, local, test
│   │   ├── model/quantization/  # none, 4bit (QLoRA), 8bit
│   │   ├── model/modifiers/     # full fine-tune, LoRA
│   │   └── data/                # sft, cpt, трансформации
│   ├── rag_pipeline/            # RAG-энкодер, FAISS, лоссы
│   │   ├── model/architecture/  # bge-m3, local, test
│   │   ├── loss/                # mnrl, triplet
│   │   └── vector_db/           # flat, hnsw
│   ├── prompts/default.yaml     # rag_qa, summarization, translation
│   └── environment/             # local, docker, k8s
│
├── src/
│   ├── api_gateway/             # Оркестратор (FastAPI)
│   ├── application/             # RAGOrchestrator, LlamaIndex-интеграция
│   ├── decoder_pipeline/
│   │   ├── core/                # DataModule, ModelBuilder, трансформации
│   │   ├── training/            # LightningModule, callbacks
│   │   ├── inference/           # Generator, ResponseCleaner
│   │   └── api/                 # REST-эндпоинты декодера
│   ├── rag_pipeline/
│   │   ├── core/                # DataModule, ModelBuilder, Pooler
│   │   ├── training/            # LightningModule, MNRL/Triplet loss
│   │   ├── indexing/            # Индексатор FAISS
│   │   ├── retrieval/           # BaseRetriever
│   │   └── api/                 # REST-эндпоинты RAG
│   ├── schemas/                 # Pydantic-схемы (main, decoder, rag)
│   ├── tools/                   # CLI: merge_lora, fetch_data, maintenance
│   └── utils/                   # hydra_utils, mlflow, vector_db, logger
│
├── scripts/
│   ├── decoder_pipeline/        # train, eval, infer, run_api
│   └── rag_pipeline/            # train, eval, index_db, infer, run_api
│
├── dags/
│   ├── llm/                     # retrain, promote, quality_control, analytics
│   └── rag/                     # fetch_data, retrain, index_db, promote, qc
│
├── notebooks/
│   └── decoder_pipeline/        # 01 EDA → 02 Baseline → 03 Prompts
│                                #   → 04 LoRA → 05 Eval → 06 Export
│
├── demo/                        # Streamlit-клиент
├── helm/                        # Helm chart для K8s
├── deploy/                      # K8s manifests, Airflow variables
├── tests/                       # pytest, полное покрытие
├── docker-compose.yml
├── Dockerfile                   # Multi-stage, uv, non-root user
├── Makefile                     # Task runner
└── pyproject.toml               # uv + группы зависимостей
```

---

## Конфигурация через Hydra

Вся конфигурация управляется **Hydra** — изменять поведение системы можно через CLI без редактирования файлов:

```bash
# Сменить архитектуру
python -m scripts.decoder_pipeline.train \
    decoder_pipeline/model/architecture=Qwen2.5-1.5B

# Включить QLoRA 4-bit
python -m scripts.decoder_pipeline.train \
    decoder_pipeline/model/quantization=4bit

# Подключить early stopping
python -m scripts.decoder_pipeline.train \
    +decoder_pipeline/training/callbacks/early_stopping@callbacks.early_stopping=default

# DDP на 4 GPU
python -m scripts.decoder_pipeline.train \
    +decoder_pipeline/training/strategy=ddp \
    training.devices=4

# Переключить RAG-лосс с MNRL на Triplet
python -m scripts.rag_pipeline.train \
    rag_pipeline/loss=triplet

# Переключить векторную базу на HNSW
python -m scripts.rag_pipeline.train \
    vector_db=hnsw

# Развернуть в окружении Docker (URL сервисов подхватятся из env)
python -m src.api_gateway.run_api \
    environment=docker
```

---

## Пайплайны

### Decoder Pipeline (LLM fine-tuning)

Задачи: **SFT** (instruction-following, перевод, суммаризация), **CPT** (continued pre-training на доменном корпусе).

**Путь данных:**
```
HF Hub / local CSV → DataFetcher → TextCleaningPipeline →
→ [ExactDedup / MinHashDedup] → ValidationTransform →
→ TokenizationTransform → FilteringTransform → SFTCollator
```

**Обучение:**
- PyTorch Lightning `CausalLMLightningModule`
- LoRA (`r=16, alpha=32`) или full fine-tune — переключается конфигом
- QLoRA 4-bit (`nf4, double_quant`) для больших моделей при ограниченной VRAM
- `bf16-mixed` precision, gradient clipping, gradient accumulation
- GenerationEvalCallback: генерирует примеры на validation прямо в процессе обучения

**После обучения:**
```bash
# Слить LoRA-адаптер в веса базовой модели и залогировать в MLflow
python -m src.tools.merge_lora pipeline_name=decoder_pipeline
```

**Инференс:**

Обученная модель подаётся через **vLLM** (OpenAI-совместимый API) — нет зависимости от Python-окружения обучения. `LLMGenerationClient` из `src/decoder_pipeline/sdk/` поддерживает батч-генерацию через `asyncio.gather` и стриминг (SSE).

---

### RAG Pipeline (энкодер + поиск)

Задачи: семантический поиск, retrieval-augmented generation, dense retrieval для QA.

**Данные:**
```
Документы (текст / пары query-passage) → IndexingDataModule →
→ Batched embedding → FAISSVectorDB (Flat или HNSW)
```

**Обучение энкодера:**
- Contrastive learning: **MNRL** (Multiple Negatives Ranking Loss) — рекомендуется для in-batch negatives
- **Triplet Loss** — для явно размеченных триплетов (anchor, positive, negative)
- Mean pooling + L2-нормализация векторов
- `RetrievalEvalCallback`: считает MRR и Recall@K на валидации в процессе обучения

**Индексация:**
```bash
python -m scripts.rag_pipeline.index_db
```

**Поиск:**
`BaseRetriever` принимает текстовый запрос → векторизует через `RAGInferenceEmbedder` → ищет в FAISS с поддержкой:
- `score_threshold` — отсечение по косинусному сходству
- `filter_metadata` — точная фильтрация по полям документов

---

### API Gateway (оркестрация)

`RAGOrchestrator` связывает два сервиса в один запрос:
```
Запрос → [RAG API: поиск top-K документов] →
→ PromptManager.render(template, context, question) →
→ [LLM API: генерация ответа] → Ответ клиенту
```

**PromptManager** читает шаблоны из `configs/prompts/default.yaml`. Продуктологи редактируют промпты без изменения Python-кода. Готовые шаблоны: `rag_qa`, `summarization`, `translation`, `telegram_welcome`.

**Мониторинг:**
- Prometheus-метрики: `gateway_requests_total`, `gateway_process_seconds` (латентность E2E)
- Rate limiting через `slowapi`
- Grafana-дашборды поднимаются через `docker-compose up prometheus grafana`

---

## Airflow DAG-и

| DAG | Расписание | Что делает |
|---|---|---|
| `llm_weekly_finetuning` | `@weekly` | SFT → merge LoRA → тест |
| `llm_promote` | вручную | Продвигает модель в `Production` в MLflow Registry |
| `llm_quality_control` | `@daily` | Батч-инференс, считает метрики, алерт в Slack |
| `llm_batch_analytics` | `@daily` | Аналитика по логам генерации |
| `rag_fetch_data` | по расписанию | Забирает новые документы |
| `rag_encoder_finetuning` | `@weekly` | Дообучение энкодера |
| `rag_index_db` | после обучения | Переиндексация FAISS |
| `rag_promote` | вручную | Продвижение энкодера |
| `rag_quality_control` | `@daily` | Проверка retrieval-метрик |

Все DAG-и используют `KubernetesPodOperator` — тяжёлые ML-зависимости не тянутся в Airflow-воркер. При ошибке прилетает алерт в Slack.

```bash
# Запустить Airflow локально в Docker
make docker_airflow
# → http://localhost:8080  (admin / admin)
```

---

## Ноутбуки

Последовательность исследований встроена в нумерованные ноутбуки:

| Ноутбук | Что делаем |
|---|---|
| `01_eda_and_tokens.ipynb` | Анализ датасета, распределение длин, токен-статистика |
| `02_generation_baseline.ipynb` | Baseline без fine-tuning: оцениваем zero-shot качество |
| `03_prompt_engineering.ipynb` | Итерация промптов, сравнение шаблонов |
| `04_peft_lora_sandbox.ipynb` | Быстрый LoRA-эксперимент прямо в ноутбуке |
| `05_evaluation_and_errors.ipynb` | Анализ ошибок, метрики, error cases |
| `06_merge_and_export.ipynb` | Слияние LoRA, экспорт в MLflow |

```bash
# Установить зависимости для экспериментов
uv pip install -e ".[experimentation]"
jupyter lab
```

---

## Docker

```bash
# Поднять всю API-инфраструктуру
make docker_api
# Сервисы:
#   :8000 — API Gateway
#   :8001 — RAG API
#   :8002 — LLM API (vLLM)
#   :8501 — Streamlit Demo
#   :9090 — Prometheus
#   :3000 — Grafana (admin/admin)

# Обучение в изолированном контейнере
make docker_train_decoder
make docker_train_rag

# Стоп
make docker_down
```

Dockerfile двухстадийный (builder + runner), использует `uv` для сборки, запускается под non-root пользователем. Группа зависимостей указывается через `INSTALL_EXTRAS` build arg — образ для обучения (`training`) отделён от образа для API (`api`).

---

## Деплой в Kubernetes

```bash
# Установить через Helm
helm install nlp-template ./helm/decoder-api-chart \
    -f helm/decoder-api-chart/values_example.yaml \
    --namespace ml-pipelines

# K8s манифесты лежат в deploy/k8s/
```

---

## Тестирование

```bash
# Все тесты
pytest

# Только конкретный модуль
pytest tests/decoder_pipeline/
pytest tests/rag_pipeline/
pytest tests/api_gateway/
pytest tests/dags/
```

Тяжёлые зависимости (модели, GPU, внешние сервисы) замокированы через `app.dependency_overrides` и `pytest.fixture`. Тесты запускаются без GPU.

---

## Зависимости

Зависимости организованы в группы — устанавливай только то, что нужно:

```bash
uv pip install -e ".[dev,training,api]"     # полная разработка
uv pip install -e ".[training]"             # только обучение
uv pip install -e ".[api]"                  # только инференс
uv pip install -e ".[tune]"                 # добавить Optuna HPO
uv pip install -e ".[demo]"                 # Streamlit
uv pip install -e ".[experimentation]"      # ноутбуки, pandas, matplotlib
```

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|---|---|---|
| `HF_TOKEN` | HuggingFace токен для gated-моделей | — |
| `MLFLOW_TRACKING_URI` | URI MLflow-сервера | `http://localhost:5000` |
| `TG_BOT_TOKEN` | Telegram Bot API токен | — |
| `LLM_MODEL` | Модель для vLLM | `facebook/opt-125m` |
| `GATEWAY_PORT` | Порт API Gateway | `8000` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

---

## Чеклист для нового проекта

- [ ] `cp .env.example .env` → заполнить `HF_TOKEN`
- [ ] Положить данные → настроить `configs/decoder_pipeline/data/source/`
- [ ] Выбрать архитектуру → `configs/decoder_pipeline/model/architecture/`
- [ ] Написать промпт → `configs/prompts/default.yaml`
- [ ] `make train_decoder` → проверить run в MLflow UI (`make mlflow`)
- [ ] `make docker_api` → отправить первый запрос на `:8000/api/v1/generate`
- [ ] Открыть `notebooks/05_evaluation_and_errors.ipynb` → начать исследование качества