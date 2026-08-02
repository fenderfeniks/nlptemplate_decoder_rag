from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


# ==============================================================================
# Базовые классы
# ==============================================================================


@dataclass
class BaseTransformConfig:
    _target_: str


@dataclass
class BaseSourceConfig:
    _target_: str


@dataclass
class BaseCollatorConfig:
    _target_: str


@dataclass
class BaseCleanerConfig:
    _target_: str


@dataclass
class BaseModifierConfig:
    _target_: str


@dataclass
class BaseStrategyConfig:
    _target_: str


@dataclass
class BaseCallbackConfig:
    _target_: str


# ==============================================================================
# Трансформации данных
# ==============================================================================


@dataclass
class InstructionTokenizationTransformConfig(BaseTransformConfig):
    _target_: str = (
        "src.decoder_pipeline.core.data.transforms.tokenization.InstructionTokenizationTransform"
    )
    text_column: str | None = None
    prompt_column: str | None = None
    target_column: str | None = None
    messages_column: str | None = None
    max_length: int = 2048
    use_chat_template: bool = False
    separator: str = "\n\n"
    num_proc: int = 4
    batch_size: int = 1000


@dataclass
class LengthFilterTransformConfig(BaseTransformConfig):
    _target_: str = "src.decoder_pipeline.core.data.transforms.filtering.LengthFilterTransform"
    max_length: int = 2048
    column: str = "input_ids"
    num_proc: int = 4


# ==============================================================================
# Источники данных, Клинеры и Коллаторы
# ==============================================================================


@dataclass
class RawDataFetcherConfig(BaseSourceConfig):
    _target_: str = "src.decoder_pipeline.core.data.fetcher.RawDataFetcher"
    source_type: str = "local"
    raw_dir: str = "${paths.data_dir}/raw"
    dataset_name: str | None = None
    file_name: str | None = None
    token: str | None = None


@dataclass
class InterleavedDataFetcherConfig(BaseSourceConfig):
    _target_: str = "src.rag_pipeline.core.data.mixers.InterleavedDataFetcher"
    seed: int = 42
    stopping_strategy: str = "first_exhausted"
    probabilities: list[float] = field(default_factory=list)
    fetchers: list[Any] = field(default_factory=list)
    imbalance_warning_ratio: float = 10.0


@dataclass
class RandomDatasetSplitterConfig:
    _target_: str = "src.decoder_pipeline.core.data.splitters.RandomDatasetSplitter"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = "${decoder_pipeline.data.seed}"  # type: ignore[assignment]


@dataclass
class InstructionDataCollatorConfig(BaseCollatorConfig):
    _target_: str = "src.decoder_pipeline.core.data.collators.InstructionDataCollator"
    max_sequence_length: int = 2048
    mask_prompt: bool = False
    response_template: str | None = None
    tokenizer: Any = None


@dataclass
class TextCleaningPipelineConfig(BaseCleanerConfig):
    _target_: str = "src.decoder_pipeline.core.data.cleaners.TextCleaningPipeline"
    cleaners: list[Any] = field(default_factory=list)


@dataclass
class DataLoaderConfig:
    batch_size: int = 8
    num_workers: int = "${system.num_workers}"  # type: ignore[assignment]
    pin_memory: bool = "${system.pin_memory}"  # type: ignore[assignment]


# ==============================================================================
# Модели и Архитектура
# ==============================================================================


@dataclass
class HFTokenizerBuilderConfig:
    _target_: str = "src.decoder_pipeline.core.models.tokenization.HFTokenizerBuilder"
    tokenizer_name: str = "${decoder_pipeline.model.architecture.model_name_or_path}"
    use_fast: bool = True
    padding_side: str = "right"
    add_eos_token: bool = False
    trust_remote_code: bool = True
    chat_template: str | None = None


@dataclass
class QuantizationConfig:
    load_in_4bit: bool | None = None
    bnb_4bit_compute_dtype: str | None = None
    bnb_4bit_quant_type: str | None = None
    bnb_4bit_use_double_quant: bool | None = None
    load_in_8bit: bool | None = None


@dataclass
class PEFTLoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class EmbeddingResizeModifierConfig(BaseModifierConfig):
    _target_: str = "src.decoder_pipeline.core.models.modifiers.EmbeddingResizeModifier"
    tokenizer: Any = "${decoder_pipeline.model.tokenizer}"  # type: ignore[assignment]


@dataclass
class PEFTModifierConfig(BaseModifierConfig):
    _target_: str = "src.decoder_pipeline.core.models.modifiers.PEFTModifier"
    gradient_checkpointing: bool = True
    is_quantized: bool = True
    lora_resume_path: str | None = None
    peft_config: PEFTLoraConfig = field(default_factory=PEFTLoraConfig)


@dataclass
class FullFTModifierConfig(BaseModifierConfig):
    _target_: str = "src.decoder_pipeline.core.models.modifiers.FullFinetuningModifier"
    gradient_checkpointing: bool = True


@dataclass
class HFModelBuilderConfig:
    _target_: str = "src.decoder_pipeline.core.models.builder.HFModelBuilder"
    _recursive_: bool = False
    model_name_or_path: str = MISSING
    auto_model_class: str = "transformers.AutoModelForCausalLM"
    cache_dir: str = "${paths.data_dir}/weights"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    quantization_config: QuantizationConfig | None = None
    modifiers: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Инференс
# ==============================================================================


@dataclass
class ResponseCleanerConfig:
    _target_: str = "src.decoder_pipeline.core.models.response_cleaner.ResponseCleaner"
    strip_prompt: bool = True
    remove_special_tokens: bool = True
    remove_markdown_blocks: bool = False
    remove_extra_spaces: bool = True
    trim_incomplete_sentence: bool = True


@dataclass
class HFTextGeneratorConfig:
    _target_: str = "src.decoder_pipeline.core.models.generator.HFTextGenerator"
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
    )
    cleaner_cfg: ResponseCleanerConfig = field(default_factory=ResponseCleanerConfig)


# ==============================================================================
# Обучение: Лоссы, Оптимизаторы и Модуль
# ==============================================================================


@dataclass
class AdamWConfig:
    _target_: str = "torch.optim.AdamW"
    _partial_: bool = True
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class CosineScheduleWithWarmupConfig:
    _target_: str = "transformers.get_cosine_schedule_with_warmup"
    _partial_: bool = True
    num_warmup_steps: int = 100


# ==============================================================================
# PyTorch Lightning: Trainer, Callbacks, Strategies
# ==============================================================================


@dataclass
class ModelCheckpointConfig(BaseCallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = "${paths.log_dir}/checkpoints"
    filename: str = "step={step}-val_loss={val_loss:.4f}"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 2
    save_last: bool = True
    auto_insert_metric_name: bool = False
    every_n_train_steps: int = 200
    save_on_train_epoch_end: bool = False


@dataclass
class LearningRateMonitorConfig(BaseCallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: str = "step"


@dataclass
class RichProgressBarConfig(BaseCallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.RichProgressBar"


@dataclass
class EarlyStoppingConfig(BaseCallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"
    min_delta: float = 0.001


@dataclass
class GenerationEvaluationCallbackConfig(BaseCallbackConfig):
    _target_: str = "src.decoder_pipeline.training.callbacks.GenerationEvaluationCallback"
    model_name: str = "${decoder_pipeline.model.architecture.mlflow_model_name}"
    num_random: int = 5
    generation_batch_size: int = 2
    mode: str = "auto"
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    fixed_samples: list[dict[str, str]] = field(default_factory=list)


@dataclass
class TrainingConfig:
    _target_: str = "pytorch_lightning.Trainer"
    default_root_dir: str = "${paths.log_dir}"
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "bf16-mixed"
    max_steps: int = 2000
    max_epochs: int = -1
    val_check_interval: int = 200
    check_val_every_n_epoch: Any = None
    accumulate_grad_batches: int = 4
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    log_every_n_steps: int = 10
    logger: Any = None
    num_sanity_val_steps: int = 2
    deterministic: bool = False
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    limit_test_batches: Any = 1.0
    callbacks: Any = field(default_factory=dict)


# ==============================================================================
# API (FastAPI)
# ==============================================================================


@dataclass
class TelegramConfig:
    bot_token: str = ""
    webhook_url: str = ""


@dataclass
class TelegramWebhookConfig:
    path: str = "/webhook/telegram"
    url: str = "${decoder_pipeline.api.domain}${decoder_pipeline.api.telegram_webhook.path}"


@dataclass
class DecoderAPIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    domain: str = "http://localhost:8000"
    concurrency_limit: int = 1
    title: str = "Decoder API"
    description: str = "API for LLM Text Generation"
    version: str = "0.1.0"
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    telegram_webhook: TelegramWebhookConfig = field(default_factory=TelegramWebhookConfig)
    generation_template: str = "rag_qa"
    generation_static_context: str = ""
    log_level: str = "INFO"


# ==============================================================================
# Корневая сборка Decoder пайплайна
# ==============================================================================


@dataclass
class DecoderPipelineConfig:
    data: Any = field(default_factory=dict)
    model: Any = field(default_factory=dict)
    optimizer: Any = field(default_factory=dict)
    scheduler: Any = field(default_factory=dict)
    training: Any = field(default_factory=dict)
    inference: HFTextGeneratorConfig = field(default_factory=HFTextGeneratorConfig)
    api: DecoderAPIConfig = field(default_factory=DecoderAPIConfig)
