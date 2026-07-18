from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PathsConfig:
    root_dir: str
    data_dir: str
    hf_cache_dir: str
    processed_data_dir: str
    log_dir: str


@dataclass
class HydraRunConfig:
    dir: str


@dataclass
class HydraJobConfig:
    chdir: bool


@dataclass
class HydraConfig:
    run: HydraRunConfig
    job: HydraJobConfig


@dataclass
class OptimizerConfig:
    _target_: str
    lr: float
    weight_decay: float


@dataclass
class ModelModuleConfig:
    _target_: str
    num_classes: int
    optimizer_cfg: OptimizerConfig
    loss_fn_cfg: Any


@dataclass
class RootDataModuleConfig:
    _target_: str
    data_cfg: Any


@dataclass
class TokenizerConfig:
    _target_: str
    tokenizer_name: str
    use_fast: bool
    padding_side: str
    add_eos_token: bool


@dataclass
class ModelBuilderConfig:
    _target_: str
    model_name_or_path: str
    cache_dir: str
    trust_remote_code: bool
    auto_model_class: str
    torch_dtype: str
    quantization_config: Any | None = None
    peft_config: Any | None = None


@dataclass
class GenerationKwargsConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float


@dataclass
class GenerationConfig:
    _target_: str
    generation_kwargs: GenerationKwargsConfig


@dataclass
class ResponseCleanerConfig:
    _target_: str
    strip_prompt: bool
    remove_special_tokens: bool
    remove_extra_spaces: bool
    trim_incomplete_sentence: bool


@dataclass
class ModelConfig:
    model_name: str
    is_causal_lm: bool
    tokenizer: TokenizerConfig
    builder: ModelBuilderConfig
    generation: GenerationConfig
    cleaner: ResponseCleanerConfig
    loss_fn: Any | None = None


@dataclass
class DataSourceConfig:
    _target_: str
    path: str
    split: str | None = None
    data_files: str | None = None
    sep: str | None = None


@dataclass
class DataCleanerPipelineConfig:
    _target_: str
    cleaners: list[Any] = field(default_factory=list)


@dataclass
class DataCollatorConfig:
    _target_: str
    max_length: int
    text_column: str
    target_column: str
    is_causal_lm: bool


@dataclass
class DataLoaderConfig:
    _target_: str
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class DataDataModuleConfig:
    _target_: str


@dataclass
class DataConfig:
    text_column: str
    target_column: str
    max_length: int
    val_split_size: float
    seed: int
    preprocessing_num_workers: int
    preprocessing_batch_size: int
    cleaner: DataCleanerPipelineConfig
    collator: DataCollatorConfig
    dataloader: DataLoaderConfig
    datamodule: DataDataModuleConfig
    source: DataSourceConfig


@dataclass
class MLFlowLoggerConfig:
    _target_: str
    experiment_name: str
    tracking_uri: str
    save_dir: str


@dataclass
class TrainerConfig:
    _target_: str
    max_epochs: int
    accelerator: str
    devices: int
    precision: str
    logger: MLFlowLoggerConfig
    callbacks: list[Any] = field(default_factory=list)


@dataclass
class RAGIndexerConfig:
    _target_: str
    documents_dir: str
    persist_dir: str
    chunk_size: int
    chunk_overlap: int
    embedding_model_name: str
    vector_dimension: int
    hnsw_m: int


@dataclass
class RAGConfig:
    documents_dir: str
    persist_dir: str
    indexer: RAGIndexerConfig
    similarity_top_k: int


@dataclass
class TelegramWebhookConfig:
    path: str
    url: str


@dataclass
class TelegramMessagesConfig:
    welcome: str
    error: str
    thinking: str


@dataclass
class TelegramConfig:
    bot_token: str
    default_use_rag: bool
    max_tokens: int
    messages: TelegramMessagesConfig


@dataclass
class APIConfig:
    host: str
    port: int
    domain: str
    title: str
    description: str
    version: str
    telegram_webhook: TelegramWebhookConfig
    cors_origins: list[str]
    telegram: TelegramConfig


@dataclass
class OptunaConfig:
    n_trials: int
    direction: str
    metric_name: str
    enable_pruning: bool
    # Можно добавить путь к БД, если хочешь сохранять историю подбора
    storage: str | None = None
    study_name: str | None = None


@dataclass
class ConfigSchema:
    seed: int
    project_name: str
    paths: PathsConfig
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    rag: RAGConfig
    api: APIConfig
    model_module: ModelModuleConfig
    datamodule: RootDataModuleConfig
    hydra: HydraConfig
    optuna: OptunaConfig | None = None
