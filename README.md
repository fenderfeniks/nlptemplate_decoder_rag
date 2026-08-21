# Industrial NLP Template — Decoder + RAG

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.2%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-89b4fa?logo=python&logoColor=white)](https://hydra.cc/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Airflow](https://img.shields.io/badge/Airflow-2.8%2B-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![FAISS](https://img.shields.io/badge/FAISS%20%2F%20Qdrant-Vector%20DB-blue)](https://github.com/facebookresearch/faiss)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-orange)](https://github.com/huggingface/peft)
[![uv](https://img.shields.io/badge/uv-fast%20packaging-DE5FE9)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/linter-ruff-black)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/coverage-~80%25-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready шаблон для задач **генерации текста (Decoder / LLM)** и **семантического поиска (RAG)**. Цель — от сырых данных до задеплоенного API-сервиса за минимальное количество шагов.

---

## Зачем этот шаблон

### Главная ценность — смоук-тест без обучения

Шаблон позволяет **проверить любую модель на ваших данных до начала обучения**. Запустите нулевой baseline через ноутбуки (zero-shot инференс, анализ промптов) — и если результаты устраивают, сервис буквально за один день можно подготовить к деплою: promote -> merge_lora -> docker_api. Обучение при этом опционально, а не обязательно.

Большинство NLP-задач — RAG-чат-боты, суммаризация, перевод, Q&A — повторяют одну и ту же инфраструктуру: загрузка данных, дообучение с LoRA, трекинг экспериментов, REST-сервис, переиндексация базы документов. Этот шаблон реализует всё это **один раз и продакшен-качественно**, оставляя только то, что специфично для задачи: данные, промпты и гиперпараметры.

**Что получаете «из коробки»:**

- Смоук-тест любой CausalLM-модели без обучения — за часы вместо дней
- Полный пайплайн дообучения LLM (CPT / SFT) с LoRA/QLoRA через PyTorch Lightning
- Полный пайплайн дообучения RAG-энкодера (contrastive learning, MNRL / Triplet loss)
- Оценка качества генерации прямо во время SFT: LLM-as-a-Judge (OpenRouter) и NLI-судья (RoBERTa)
- Векторная база FAISS (Flat / HNSW) или Qdrant — переключение одной строкой в конфиге
- Три REST-микросервиса (API Gateway, RAG API, Decoder API) с мониторингом Prometheus + Grafana
- Транспортировка весов через манифест: local, S3, HF Hub
- Airflow DAG-и для регулярного переобучения, индексации и quality control
- Helm-чарт для деплоя в Kubernetes
- Streamlit-демо и Telegram-бот для быстрой проверки результата
- Jupyter-ноутбуки: от EDA до экспорта модели
- pytest ~80% покрытие кода, linting через Ruff

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                  CLIENT / TG BOT / DEMO                 │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│              API Gateway  (FastAPI, :8000)               │
│   RAGOrchestrator: запрос -> RAG API -> промпт -> LLM API  │
│   PromptManager: шаблоны из YAML, без хардкода          │
│   Middleware: CORS, rate-limit, Prometheus-метрики       │
└────────────┬────────────────────────┬───────────────────┘
             │ HTTP                   │ HTTP
┌────────────▼──────────┐  ┌──────────▼──────────────────┐
│  RAG API  (FastAPI,   │  │  Decoder API  (FastAPI,      │
│  :8001)               │  │  :8003) -> vLLM (:8002)      │
│  Запрос -> embedding   │  │  Принимает промпт + docs     │
│  -> поиск по FAISS     │  │  Отдаёт ответ в Gateway      │
│  -> возврат документов │  │  OpenAI-compatible /v1       │
└───────────────────────┘  └──────────────────────────────┘

ОБУЧЕНИЕ (batch jobs):
  scripts/decoder_pipeline/train.py  — CPT/SFT + LoRA -> MLflow
  scripts/rag_pipeline/train.py      — Contrastive -> MLflow
  scripts/rag_pipeline/index_db.py   — Индексация FAISS + обновление манифеста
  -> трекинг:    MLflow (единая БД, раздельные experiment name)
  -> промоут:    src/tools/promote.py  (LoRA из mlruns -> storage + манифест)
  -> слияние:    src/tools/merge_lora.py (LoRA + base -> merged model)
  -> оркестрация: Airflow DAGs (KubernetesPodOperator)
  -> деплой:     Helm chart
```

---

## Структура проекта

```
.
├── configs/                          # Вся конфигурация (Hydra)
│   ├── main.yaml                     # Точка входа, собирает пайплайн
│   ├── decoder_pipeline/
│   │   ├── data/
│   │   │   ├── sft.yaml              # SFT: prompt_column + target_column + separator
│   │   │   ├── cpt.yaml              # CPT: text_column, packing включён по умолчанию
│   │   │   └── transforms/packing/   # SequencePackingTransform — чанкование для CPT
│   │   ├── model/
│   │   │   ├── architecture/         # Qwen2.5-1.5B, Qwen3-4B-Instruct, phi-4-mini, local
│   │   │   ├── quantization/         # none, 4bit (QLoRA), 8bit
│   │   │   └── modifiers/finetuning/ # LoRA или full fine-tune
│   │   └── training/                 # Trainer, callbacks (GenerationEvaluationCallback)
│   ├── rag_pipeline/
│   │   ├── data/
│   │   │   └── transforms/chunking/  # OverlappingChunkingTransform (по символам + overlap)
│   │   ├── model/architecture/       # bge-m3, local
│   │   ├── loss/                     # mnrl, triplet
│   │   └── vector_db/                # flat, hnsw, qdrant
│   ├── evaluation/
│   │   └── judge/                    # nli (RoBERTa-like), openrouter (LLM-as-a-Judge)
│   ├── storage/source/               # local, s3, hf_hub
│   ├── manifest/default.yaml         # URI манифеста модели
│   ├── prompts/default.yaml          # rag_qa, summarization, translation, telegram_welcome
│   └── environment/                  # local, docker, k8s
│
├── src/
│   ├── api_gateway/                  # Оркестратор (FastAPI)
│   ├── application/
│   │   ├── orchestrator.py           # RAGOrchestrator: RAG API -> PromptManager -> LLM
│   │   └── llamaindex_ext.py         # LlamaIndex-интеграция
│   ├── pipelines/
│   │   ├── base/                     # Общие строительные блоки (ModelBuilder, DataModule)
│   │   │   └── core/data/transforms/ # Очистка, дедупликация (MinHash), валидация, фильтрация
│   │   ├── decoder/
│   │   │   ├── core/data/
│   │   │   │   ├── collators.py      # InstructionDataCollator: маскирование промпта по prompt_len
│   │   │   │   └── transforms/packing.py  # SequencePackingTransform (конкатенация + нарезка)
│   │   │   ├── training/
│   │   │   │   ├── module.py         # CausalLMLightningModule (CPT/SFT, perplexity)
│   │   │   │   └── callbacks.py      # GenerationEvaluationCallback (ROUGE, BLEU, Judge)
│   │   │   ├── inference/            # Generator, ResponseCleaner, LLMGenerationClient
│   │   │   └── api/                  # REST-эндпоинты декодера
│   │   └── rag/
│   │       ├── core/data/transforms/
│   │       │   └── chunking.py       # OverlappingChunkingTransform (символы, overlap)
│   │       ├── training/
│   │       │   ├── module.py         # RAGLightningModule
│   │       │   ├── losses.py         # MNRL, Triplet loss
│   │       │   └── callbacks.py      # RetrievalEvaluationCallback (Recall@K, NDCG@K, MRR)
│   │       ├── inference/embedder.py # RAGInferenceEmbedder
│   │       ├── indexing/indexer.py   # FAISSIndexer
│   │       ├── retrieval/retriever.py # BaseRetriever (score_threshold, filter_metadata)
│   │       └── api/                  # REST-эндпоинты RAG
│   ├── schemas/                      # Pydantic-схемы (main, decoder, rag, nli, evaluation)
│   ├── tools/
│   │   ├── promote.py                # LoRA из MLflow -> storage + манифест
│   │   ├── merge_lora.py             # LoRA + base -> merged model -> storage + манифест
│   │   ├── fetch_data.py             # Загрузка данных
│   │   ├── maintenance.py            # Очистка кэша
│   │   ├── batch_analytics.py        # Аналитика логов генерации
│   │   ├── storage/                  # local.py, s3.py, hf_hub.py, router.py
│   │   └── evaluation/judges/        # BaseJudge, LLMJudge (OpenRouter), NLIJudge (RoBERTa)
│   ├── vector_store/                 # faiss_store.py, qdrant_store.py, lsh.py, base.py
│   ├── tg_bot/                       # Telegram-бот (handlers, keyboards, webhook/polling)
│   └── utils/                        # mlflow.py, hydra_utils.py, torch_utils, logger
│
├── scripts/
│   ├── decoder_pipeline/             # train.py, eval.py, infer.py, run_api.py
│   ├── rag_pipeline/                 # train.py, eval.py, index_db.py, infer.py, run_api.py
│   ├── api_gateway/run_api.py
│   ├── prepare_artifacts.py          # promote + merge_lora в один шаг
│   └── download_artifacts.py         # Скачивание весов по манифесту (init container)
│
├── dags/
│   ├── llm/                          # llm_retrain, llm_promote, llm_quality_control, llm_batch_analytics
│   └── rag/                          # rag_fetch_data, rag_retrain, rag_index_db, rag_promote, rag_quality_control, rag_batch_analytics
│
├── notebooks/
│   └── decoder_pipeline/             # 01 EDA -> 02 Baseline -> 03 Prompts
│                                     #   -> 04 LoRA -> 05 Eval -> 06 Export
│                                     # TODO: RAG-ноутбуки
│
├── demo/                             # Streamlit-клиент
├── helm/decoder-api-chart/           # Helm chart для K8s (конфиги прокидываются через values)
├── deploy/                           # K8s manifests, Airflow variables
├── tests/                            # pytest, ~80% покрытие
├── docker-compose.yml
├── Dockerfile                        # Multi-stage, uv, non-root user; INSTALL_EXTRAS для групп
├── Makefile                          # Task runner
└── pyproject.toml                    # uv + группы зависимостей
```

---

## Быстрый старт

### 1. Установка

```bash
git clone <your-repo-url> && cd nlp-template
make install          # uv venv + uv pip install -e ".[dev,training,api]"
cp .env.example .env  # заполни HF_TOKEN и MLFLOW_TRACKING_URI
```

Минимально необходимые переменные в `.env`:

```dotenv
HF_TOKEN=hf_...                       # для gated-моделей (Llama, Phi, Qwen и т.д.)
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 2. Смоук-тест без обучения (рекомендуется с этого начинать)

```bash
# Открой ноутбуки и проверь zero-shot качество на своих данных
jupyter lab
# notebooks/decoder_pipeline/02_generation_baseline.ipynb
# notebooks/decoder_pipeline/03_prompt_engineering.ipynb
```

Если zero-shot результаты устраивают — дообучение не нужно. Можно сразу переходить к деплою.

### 3. Подготовь данные

Поддерживаются HuggingFace Hub (по умолчанию для SFT), локальные файлы (по умолчанию для CPT) и Kaggle:

```yaml
# configs/decoder_pipeline/data/source/hf.yaml
dataset_name: "your-org/your-dataset"
split: "train"
```

Или переключись через CLI: `decoder_pipeline/data/source=local`

### 4. Выбери архитектуру

```bash
# Готовые конфиги — configs/decoder_pipeline/model/architecture/
# Qwen2.5-1.5B    — компактная, 8 GB VRAM
# Qwen3-4B-Instruct-2507 — актуальный instruct-вариант
# phi-4-mini-instruct    — Microsoft SSM-модель
# local.yaml             — любая локальная модель

# Для RAG — configs/rag_pipeline/model/architecture/
# bge-m3.yaml   — BAAI/bge-m3 (мультиязычный)
# local.yaml    — любой локальный энкодер
```

### 5. Запусти обучение

```bash
# SFT (дообучение LLM на инструкциях)
make train_decoder

# CPT (continued pre-training на доменном корпусе)
make train_decoder ARGS="decoder_pipeline/data=cpt"

# Дообучение RAG-энкодера
make train_rag

# Переключить модель
make train_decoder ARGS="decoder_pipeline/model/architecture=Qwen3-4B-Instruct-2507"

# QLoRA 4-bit для экономии VRAM
make train_decoder ARGS="decoder_pipeline/model/quantization=4bit"

# Больше шагов
make train_decoder ARGS="training.max_steps=5000"
```

### 6. Проверь результаты в MLflow

```bash
make mlflow   # -> http://127.0.0.1:5000
```

Каждый прогон автоматически логирует метрики, гиперпараметры и LoRA-адаптер.

### 7. Перевод модели в продакшн

```bash
# Шаг A: promote — переносит лучший LoRA из mlruns в storage + создаёт манифест
python -m src.tools.promote pipeline_name=decoder_pipeline

# Шаг Б: merge_lora — сливает LoRA с базовой моделью -> готовая модель в storage
python -m src.tools.merge_lora pipeline_name=decoder_pipeline

# Или оба шага за раз:
python -m scripts.prepare_artifacts pipeline_name=decoder_pipeline

# Для RAG — сначала индексация (она же обновляет манифест с путём к БД):
python -m scripts.rag_pipeline.index_db
```

### 8. Подними API

```bash
# Весь стек одной командой
make docker_api
# :8000 — API Gateway   :8001 — RAG API
# :8002 — vLLM (LLM)   :8501 — Streamlit Demo
# :9090 — Prometheus    :3000 — Grafana (admin/admin)

# Или локально
make api_gateway   # -> http://localhost:8000
make api_rag       # -> http://localhost:8001
```

---

## Пайплайны

### Decoder Pipeline: CPT и SFT

Поддерживает два режима дообучения — режим определяется через конфиг данных (`task: cpt` или `task: sft`):

#### CPT — Continued Pre-Training

Данные подаются как поток текстов без разметки. Перед обучением применяется `SequencePackingTransform`: все последовательности конкатенируются и нарезаются на блоки фиксированного размера `packing_chunk_size` токенов. Это устраняет паддинг и существенно повышает эффективность использования GPU.

```
Текст -> очистка -> токенизация -> конкатенация -> нарезка на блоки по N токенов
```

**Метрики в MLflow (CPT):** `train_loss`, `val_loss`, `train_perplexity`, `val_perplexity`

#### SFT — Supervised Fine-Tuning

Данные подаются как пары `(prompt, target)`. Формат последовательности:

```
[system_tokens] + prompt + separator + target
```

`InstructionDataCollator` маскирует промпт в лейблах (`-100`) — loss считается только по ответу модели. Разделитель `separator` задаётся в конфиге (например: `"Перевод: "`, `"\n\n"` и т.д.). Системные токены подключаются через `use_chat_template=true` или через `messages_column` для диалогового формата.

```
Пары (prompt, target) -> очистка -> дедупликация (MinHash) -> токенизация -> маскирование промпта
```

**Метрики в MLflow (SFT):** `train_loss`, `val_loss`, `val_sacrebleu`, `val_rouge1`, `val_rougeL`

#### Judge-оценка при SFT (опционально)

`GenerationEvaluationCallback` запускает оценку генерации на валидации. Поддерживаются два судьи — подключаются через `configs/evaluation/judge/`:

- **LLM-as-a-Judge** (`judge/openrouter`) — вызывает внешнюю LLM через OpenRouter (OpenAI-compatible). Настраиваемые флаги: `return_score`, `return_verdict`, `return_reasoning`. Судья создаётся лениво — не занимает VRAM во время обучения.
- **NLI-Judge** (`judge/nli`) — локальная NLI-модель (RoBERTa-large-mnli и аналоги). `premise = reference`, `hypothesis = response` -> entailment score -> `EvalResult.score ∈ [0, 1]`.

**Дополнительные метрики при включённом judge:** `val_judge_score`, `val_judge_verdict`

---

### RAG Pipeline: Энкодер + Поиск

#### Подготовка данных

Для индексации применяется `OverlappingChunkingTransform` — нарезка на уровне слов с перекрытием:

```
Документы -> очистка -> чанкинг (chunk_size симв., chunk_overlap симв.) -> эмбеддинг -> FAISS
```

> Обрати внимание: `chunk_size` и `chunk_overlap` задаются в **символах**, не токенах. Рекомендуется калибровать эмпирически под конкретный токенизатор (≈ chunk_size=400 при max_length=128).

#### Обучение энкодера

- **MNRL** (Multiple Negatives Ranking Loss) — рекомендуется для in-batch negatives, пары `(query, passage)`
- **Triplet Loss** — для явно размеченных триплетов `(anchor, positive, negative)`
- Mean pooling + L2-нормализация

**Метрики в MLflow (RAG):** `val_recall_10`, `val_ndcg_10`, `val_mrr` — считает `RetrievalEvaluationCallback` в конце каждой валидационной эпохи.

#### Векторное хранилище

Переключение одной строкой в конфиге (`vector_db: flat` / `vector_db: hnsw` / `vector_db: qdrant`):

- **FAISS Flat** — точный поиск, рекомендуется до ~1M векторов
- **FAISS HNSW** — приближённый поиск, миллионы документов
- **Qdrant** — внешний сервер, нативная фильтрация по метаданным

`BaseRetriever` поддерживает `score_threshold` (отсечение по косинусному сходству) и `filter_metadata` (точная фильтрация по полям документов).

---

### Транспортировка моделей: promote -> merge_lora -> инференс

```
Обучение завершено
      │
      ▼
 log_lora_to_mlflow()        — LoRA-адаптер логируется как MLflow-артефакт
      │
      ▼
 promote.py                  — Сравнение Staging vs Production по val_loss
  ├─ Staging лучше           -> обновить алиас Production в MLflow Registry
  │                          -> скачать LoRA из mlruns
  │                          -> загрузить в storage (local / S3 / HF Hub)
  │                          -> обновить манифест (load_type=lora)
  └─ Staging хуже            -> промоут отменён, Production не меняется
      │
      ▼
 merge_lora.py               — Слияние LoRA + base model
  ├─ Скачать LoRA из storage
  ├─ Загрузить base model
  ├─ PeftModel.merge_and_unload()
  ├─ Сохранить merged model в storage
  └─ Обновить манифест (load_type=merged)
      │
      ▼
 Инференс загружает model по манифесту
 (manifest_uri из конфига -> storage router -> local / S3 / HF Hub)
```

**Для RAG pipeline** манифест дополнительно содержит путь к векторной БД. Индексация (`index_db.py`) обязательно должна быть выполнена до деплоя RAG API — она добавляет `vector_db_uri` в манифест.

> ⚠️ Логика для `full_finetuning` (без LoRA) реализована частично — использование возможно, но поведение promote/merge может отличаться от ожидаемого.

> ⚠️ Логика для локального тестирования полного пайплайна end-to-end — в TODO.

---

### Три микросервиса инференса

```
Запрос пользователя
      │
      ▼
 API Gateway (:8000)         — RAGOrchestrator
  ├─ POST /api/v1/chat
  ├─ Запрос к RAG API: GET /search?query=...
  │      RAG API (:8001)     — принимает query, подключается к FAISS, возвращает top-K документов
  ├─ PromptManager.render(template, context=docs, question=query)
  │      (шаблоны из configs/prompts/default.yaml)
  ├─ Запрос к Decoder API / vLLM (:8002): POST /v1/chat/completions
  │      Decoder API (:8003) — принимает промпт, генерирует ответ через vLLM
  └─ Ответ клиенту
```

**API Gateway** управляет оркестрацией, применяет CORS и rate limiting (`slowapi`), собирает Prometheus-метрики (`gateway_requests_total`, `gateway_process_seconds`).

**PromptManager** читает шаблоны из YAML — `rag_qa`, `summarization`, `translation`, `telegram_welcome`. Продуктологи меняют промпты без правки Python-кода.

**Decoder API** — тонкий слой поверх vLLM с OpenAI-compatible `/v1/`. Переключение между моделями без изменения кода.

---

## Конфигурация через Hydra

Вся конфигурация управляется **Hydra** — поведение меняется через CLI без редактирования файлов. В каждом скрипте выстроена защита от ошибки в `pipeline_name`: при запуске `scripts/decoder_pipeline/train.py` принудительно подставляется `decoder_pipeline`, при запуске `scripts/rag_pipeline/train.py` — `rag_pipeline`.

```bash
# Сменить архитектуру
python -m scripts.decoder_pipeline.train \
    decoder_pipeline/model/architecture=Qwen2.5-1.5B

# Включить QLoRA 4-bit
python -m scripts.decoder_pipeline.train \
    decoder_pipeline/model/quantization=4bit

# Переключить на CPT
python -m scripts.decoder_pipeline.train \
    decoder_pipeline/data=cpt

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

# Переключить векторную базу на Qdrant
python -m scripts.rag_pipeline.train \
    vector_db=qdrant

# Задеплоить в Docker-окружение
python -m src.api_gateway.run_api \
    environment=docker
```

---

## MLflow: единая БД, раздельные эксперименты

Все пайплайны пишут в одну MLflow-базу данных (`MLFLOW_TRACKING_URI`), но в разные эксперименты:

| Пайплайн | Experiment Name | Метрики |
|---|---|---|
| CPT | `decoder_cpt` | `train_loss`, `val_loss`, `train_perplexity`, `val_perplexity` |
| SFT | `decoder_sft` | `train_loss`, `val_loss`, `val_sacrebleu`, `val_rouge1`, `val_rougeL`, `val_judge_score`* |
| RAG | `rag_encoder` | `val_recall_10`, `val_ndcg_10`, `val_mrr` |

*только при подключённом judge

---

## Airflow DAG-и

| DAG | Расписание | Что делает |
|---|---|---|
| `llm_retrain` | `@weekly` | CPT/SFT -> log_lora -> Staging |
| `llm_promote` | вручную | Staging -> Production сравнение, обновление манифеста |
| `llm_quality_control` | `@daily` | Батч-инференс -> метрики -> алерт в Slack при деградации |
| `llm_batch_analytics` | `@daily` | Аналитика по логам генерации |
| `rag_fetch_data` | по расписанию | Забирает новые документы из источника |
| `rag_retrain` | `@weekly` | Contrastive fine-tuning энкодера |
| `rag_index_db` | после обучения | Переиндексация FAISS + обновление манифеста |
| `rag_promote` | вручную | Продвижение энкодера в Production |
| `rag_quality_control` | `@daily` | Проверка retrieval-метрик |
| `rag_batch_analytics` | `@daily` | Аналитика поисковых запросов |

Все DAG-и используют `KubernetesPodOperator` — тяжёлые ML-зависимости не попадают в Airflow-воркер. При ошибке — алерт в Slack.

```bash
make docker_airflow   # -> http://localhost:8080  (admin / admin)
```

---

## Docker

Образы разделены по задачам — нет одного большого образа на 20 ГБ:

- Лёгкие образы для FastAPI-сервисов (`INSTALL_EXTRAS=api`)
- Тяжёлые образы для обучения (`INSTALL_EXTRAS=training`) с попыткой минимизации лишних зависимостей
- vLLM использует официальный образ `vllm/vllm-openai:latest`

Dockerfile двухстадийный (builder + runner), пакетный менеджер `uv`, запуск под non-root пользователем.

```bash
# Весь API-стек
make docker_api

# Обучение в изолированном контейнере
make docker_train_decoder
make docker_train_rag

# Стоп
make docker_down
```

---

## Деплой в Kubernetes

```bash
helm install nlp-template ./helm/decoder-api-chart \
    -f helm/decoder-api-chart/values_example.yaml \
    --namespace ml-pipelines
```

Конфиги прокидываются через Helm values в K8s. `deploy/k8s/` содержит манифесты для деплоя без Helm.

> ⚠️ Helm-чарт написан, но полноценное тестирование в кластере не проводилось.

---

## Мониторинг

Prometheus собирает метрики с API Gateway и RAG API, Grafana строит дашборды:

```bash
docker compose up -d prometheus grafana
# Grafana -> http://localhost:3000  (admin / admin)
```

Ключевые метрики: `gateway_requests_total`, `gateway_process_seconds` (E2E латентность), `rag_search_latency_seconds`.

---

## Ноутбуки

| Ноутбук | Что делаем |
|---|---|
| `01_eda_and_tokens.ipynb` | Анализ датасета, распределение длин, токен-статистика |
| `02_generation_baseline.ipynb` | **Zero-shot смоук-тест** — оцениваем качество без обучения |
| `03_prompt_engineering.ipynb` | Итерация промптов, сравнение шаблонов |
| `04_peft_lora_sandbox.ipynb` | Быстрый LoRA-эксперимент прямо в ноутбуке |
| `05_evaluation_and_errors.ipynb` | Анализ ошибок, метрики, error cases |
| `06_merge_and_export.ipynb` | Слияние LoRA, экспорт в MLflow |

> TODO: Ноутбуки для RAG pipeline (EDA векторов, анализ retrieval-качества, дебаггинг поиска).

```bash
uv pip install -e ".[experimentation]"
jupyter lab
```

---

## Тестирование и качество кода

```bash
# Все тесты (тяжёлые зависимости замоканы — GPU не нужен)
pytest

# По модулям
pytest tests/pipelines/decoder/
pytest tests/pipelines/rag/
pytest tests/application/
pytest tests/dags/

# Линтинг
ruff check src/ tests/
```

Покрытие тестами — ~80%. Тяжёлые зависимости (GPU, модели, внешние сервисы) замокированы через `pytest.fixture` и `app.dependency_overrides`.

---

## Зависимости

```bash
uv pip install -e ".[dev,training,api]"      # полная разработка
uv pip install -e ".[training]"              # только обучение
uv pip install -e ".[api]"                   # только инференс/API
uv pip install -e ".[data]"                  # подготовка данных (datasets, datasketch, DVC)
uv pip install -e ".[tune]"                  # Optuna HPO
uv pip install -e ".[demo]"                  # Streamlit
uv pip install -e ".[experimentation]"       # ноутбуки, pandas, matplotlib
```

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|---|---|---|
| `HF_TOKEN` | HuggingFace токен для gated-моделей | — |
| `MLFLOW_TRACKING_URI` | URI MLflow-сервера | `http://localhost:5000` |
| `LLM_API_URL` | URL vLLM-сервера | `http://localhost:8000/v1` |
| `RAG_API_URL` | URL RAG API | `http://localhost:8001` |
| `TG_BOT_TOKEN` | Telegram Bot API токен | — |
| `OPENROUTER_API_KEY` | Ключ для LLM-as-a-Judge через OpenRouter | — |
| `GATEWAY_PORT` | Порт API Gateway | `8000` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

---

## Чеклист для нового проекта

- [ ] `cp .env.example .env` -> заполнить `HF_TOKEN`, `MLFLOW_TRACKING_URI`
- [ ] Запустить `notebooks/02_generation_baseline.ipynb` — оценить zero-shot качество
- [ ] Настроить `configs/decoder_pipeline/data/source/` под свои данные
- [ ] Выбрать архитектуру -> `configs/decoder_pipeline/model/architecture/`
- [ ] Написать промпт -> `configs/prompts/default.yaml`
- [ ] `make train_decoder` -> проверить run в MLflow UI (`make mlflow`)
- [ ] `python -m scripts.prepare_artifacts pipeline_name=decoder_pipeline` — promote + merge
- [ ] `python -m scripts.rag_pipeline.index_db` — индексация (если используется RAG)
- [ ] `make docker_api` -> отправить первый запрос на `:8000/api/v1/chat`
- [ ] Открыть `notebooks/05_evaluation_and_errors.ipynb` -> анализ качества
