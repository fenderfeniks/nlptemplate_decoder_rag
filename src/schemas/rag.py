# src/schemas/rag.py
from dataclasses import dataclass, field
from typing import Any


# ==============================================================================
# Базовые классы и Трансформации данных
# ==============================================================================


@dataclass
class BaseTransformConfig:
    _target_: str


@dataclass
class OverlappingChunkingTransformConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.chunking.OverlappingChunkingTransform"
    text_column: str = "text"
    chunk_size: int = 500
    chunk_overlap: int = 50
    separator: str = " "
    num_proc: int = 4
    batch_size: int = 1000


@dataclass
class ExactDeduplicationTransformConfig(BaseTransformConfig):
    _target_: str = (
        "src.rag_pipeline.core.data.transforms.deduplication.ExactDeduplicationTransform"
    )
    target_columns: list[str] = field(default_factory=lambda: ["text"])
    num_proc: int = 4
    column_separator: str = " "


@dataclass
class MinHashDeduplicationTransformConfig(BaseTransformConfig):
    _target_: str = (
        "src.rag_pipeline.core.data.transforms.deduplication.MinHashDeduplicationTransform"
    )
    target_columns: list[str] = field(default_factory=lambda: ["text"])
    num_perm: int = 128
    threshold: float = 0.85
    ngram_size: int = 5
    num_proc: int = 4
    column_separator: str = " "


@dataclass
class LengthFilterTransformConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.filtering.LengthFilterTransform"
    max_length: int = 2048
    column: str = "input_ids"
    num_proc: int = 4


@dataclass
class MetadataInjectorTransformConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.metadata.MetadataInjectorTransform"
    text_column: str = "text"
    metadata_column: str = "metadata"
    template: str = "{meta_string}\n\n{text}"
    num_proc: int = 4
    batch_size: int = 1000


@dataclass
class CleaningTransformConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.validation.CleaningTransform"
    columns_to_clean: list[str] = field(default_factory=list)
    num_proc: int = 4
    batch_size: int = 1000
    pipeline: list[Any] | None = None


@dataclass
class RAGTokenizationContrastiveConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.tokenization.RAGTokenizationTransform"
    mode: str = "contrastive"
    query_column: str = "query"
    positive_column: str = "positive_doc"
    negative_column: str = "negative_doc"
    max_length: int = 512
    num_proc: int = 4
    batch_size: int = 1000
    empty_doc_placeholder: str = ""


@dataclass
class RAGTokenizationIndexingConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.tokenization.RAGTokenizationTransform"
    mode: str = "indexing"
    text_column: str = "text"
    max_length: int = 512
    num_proc: int = 4
    batch_size: int = 1000


@dataclass
class ValidationContrastiveConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.validation.ValidationTransform"
    mode: str = "contrastive"
    query_column: str = "query"
    positive_column: str = "positive_doc"
    negative_column: str = "negative_doc"
    num_proc: int = 4
    batch_size: int = 1000


@dataclass
class ValidationIndexingConfig(BaseTransformConfig):
    _target_: str = "src.rag_pipeline.core.data.transforms.validation.ValidationTransform"
    mode: str = "indexing"
    text_column: str = "text"
    num_proc: int = 4
    batch_size: int = 1000


# ==============================================================================
# Источники данных, Клинеры и Коллаторы
# ==============================================================================


@dataclass
class RandomDatasetSplitterConfig:
    _target_: str = "src.rag_pipeline.core.data.splitters.RandomDatasetSplitter"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42


@dataclass
class BaseSourceConfig:
    _target_: str


@dataclass
class RawDataFetcherConfig(BaseSourceConfig):
    _target_: str = "src.rag_pipeline.core.data.fetcher.RawDataFetcher"
    source_type: str = "local"
    raw_dir: str = "${paths.data_dir}/raw"
    dataset_name: str | None = None
    file_name: str | None = None
    token: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterleavedDataFetcherConfig(BaseSourceConfig):
    _target_: str = "src.rag_pipeline.core.data.mixers.InterleavedDataFetcher"
    seed: int = 42
    stopping_strategy: str = "first_exhausted"
    probabilities: list[float] = field(default_factory=list)
    fetchers: list[Any] = field(default_factory=list)
    imbalance_warning_ratio: float = 10.0


@dataclass
class BaseCollatorConfig:
    _target_: str


@dataclass
class ContrastiveDataCollatorConfig(BaseCollatorConfig):
    _target_: str = "src.rag_pipeline.core.data.collators.ContrastiveDataCollator"
    max_length: int = 512


@dataclass
class IndexingDataCollatorConfig(BaseCollatorConfig):
    _target_: str = "src.rag_pipeline.core.data.collators.IndexingDataCollator"
    max_length: int = 512
    text_column: str = "text"
    metadata_column: str = "metadata"


@dataclass
class BaseCleanerConfig:
    _target_: str


@dataclass
class RegexCleanerConfig(BaseCleanerConfig):
    _target_: str = "src.rag_pipeline.core.data.cleaners.RegexCleaner"
    pattern: str = ""
    replacement: str = ""


@dataclass
class TextCleaningPipelineConfig:
    _target_: str = "src.rag_pipeline.core.data.cleaners.TextCleaningPipeline"
    cleaners: list[Any] = field(default_factory=list)


# ==============================================================================
# Модели и Архитектура
# ==============================================================================


@dataclass
class HFTokenizerBuilderConfig:
    _target_: str = "src.rag_pipeline.core.models.tokenization.HFTokenizerBuilder"
    tokenizer_name: str = "BAAI/bge-m3"
    use_fast: bool = True
    padding_side: str = "right"
    add_eos_token: bool = False
    trust_remote_code: bool = True
    chat_template: str | None = None
    base_model_uri: str | None = None


@dataclass
class PoolerConfig:
    _target_: str = "src.rag_pipeline.core.models.pooling.Pooler"
    pooling_mode: str = "mean"
    normalize: bool = True


@dataclass
class EmbeddingResizeModifierConfig:
    _target_: str = "src.rag_pipeline.core.models.modifiers.EmbeddingResizeModifier"
    pad_to_multiple_of: int = 8


@dataclass
class PEFTModifierConfig:
    _target_: str = "src.rag_pipeline.core.models.modifiers.PEFTModifier"
    gradient_checkpointing: bool = True
    is_quantized: bool = True
    lora_resume_path: str | None = None
    peft_config: dict[str, Any] = field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"],
        }
    )


@dataclass
class HFModelBuilderConfig:
    _target_: str = "src.rag_pipeline.core.models.builder.HFModelBuilder"
    model_name_or_path: str = "BAAI/bge-m3"
    auto_model_class: str = "transformers.AutoModel"
    cache_dir: str | None = "${paths.data_dir}/weights"
    quantization_config: dict[str, Any] | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    rope_scaling: dict[str, Any] | None = None
    modifiers: dict[str, Any] | None = None


# ==============================================================================
# Векторный поиск и Инференс
# ==============================================================================


@dataclass
class VectorDBFiltersConfig:
    active_only: dict[str, str] = field(default_factory=lambda: {"status": "active"})
    current_year: dict[str, int] = field(default_factory=lambda: {"year": 2024})
    trusted_sources: dict[str, Any] = field(
        default_factory=lambda: {"is_verified": True, "source_type": "official"}
    )


@dataclass
class BaseRetrieverConfig:
    _target_: str = "src.rag_pipeline.retrieval.retriever.BaseRetriever"


@dataclass
class RAGInferenceEmbedderConfig:
    _target_: str = "src.rag_pipeline.inference.embedder.RAGInferenceEmbedder"
    device: str = "cuda"
    precision: str = "bf16"
    max_length: int = 512


@dataclass
class KnowledgeBaseIndexerConfig:
    _target_: str = "src.rag_pipeline.indexing.indexer.KnowledgeBaseIndexer"
    device: str = "cuda"
    precision: str = "bf16"
    push_batch_size: int = 10000


# ==============================================================================
# Обучение: Лоссы, Оптимизаторы и Модуль
# ==============================================================================


@dataclass
class TripletLossWrapperConfig:
    _target_: str = "src.rag_pipeline.losses.TripletLossWrapper"
    margin: float = 1.0
    p: float = 2.0


@dataclass
class MultipleNegativesRankingLossConfig:
    _target_: str = "src.rag_pipeline.losses.MultipleNegativesRankingLoss"
    scale: float = 20.0


@dataclass
class AdamWConfig:
    _target_: str = "torch.optim.AdamW"
    _partial_: bool = True
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class CosineScheduleWithWarmupConfig:
    _target_: str = "transformers.get_cosine_schedule_with_warmup"
    _partial_: bool = True
    num_warmup_steps: int = 100


@dataclass
class RAGLightningModuleConfig:
    _target_: str = "src.rag_pipeline.training.module.RAGLightningModule"
    optimizer_cfg: Any = field(default_factory=AdamWConfig)
    scheduler_cfg: Any | None = field(default_factory=CosineScheduleWithWarmupConfig)


# ==============================================================================
# PyTorch Lightning: training, Callbacks, Strategies
# ==============================================================================


@dataclass
class ModelCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = "${paths.log_dir}/checkpoints"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 2
    save_last: bool = True
    auto_insert_metric_name: bool = False
    filename: str = "step={step}-val_loss={val_loss:.4f}"
    every_n_train_steps: Any = "${decoder_pipeline.training.val_check_interval}"
    save_on_train_epoch_end: bool = False


@dataclass
class EarlyStoppingConfig:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"
    min_delta: float = 0.001


@dataclass
class LearningRateMonitorConfig:
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: str = "step"


@dataclass
class RetrievalEvaluationCallbackConfig:
    _target_: str = "src.rag_pipeline.training.callbacks.RetrievalEvaluationCallback"
    top_k: int = 10


@dataclass
class RichProgressBarConfig:
    _target_: str = "pytorch_lightning.callbacks.RichProgressBar"


@dataclass
class DeepSpeedStrategyConfig:
    _target_: str = "pytorch_lightning.strategies.DeepSpeedStrategy"
    stage: int = 2
    offload_optimizer: bool = False
    allgather_bucket_size: float = 2e8
    reduce_bucket_size: float = 2e8


@dataclass
class DDPStrategyConfig:
    _target_: str = "pytorch_lightning.strategies.DDPStrategy"
    find_unused_parameters: bool = False


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
    check_val_every_n_epoch: int | None = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    log_every_n_steps: int = 10
    logger: Any = "${logger.pylightning}"
    num_sanity_val_steps: int = 2
    deterministic: bool = False


# ==============================================================================
# API (FastAPI)
# ==============================================================================


@dataclass
class FastAPIConfig:
    host: str = "0.0.0.0"
    port: Any = "${oc.env:API_PORT,8001}"
    domain: Any = "${oc.env:API_DOMAIN,'http://localhost:8001'}"
    concurrency_limit: int = 1
    title: Any = "${oc.env:PROJECT_NAME,'Industrial NLP Template API'}"
    description: str = "NLP API for Text Generation and LLM Inference"
    version: Any = "${oc.env:PROJECT_VERSION,'0.1.0'}"
    generation_template: str = "rag_qa"
    generation_static_context: str = ""
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )


# ==============================================================================
# Корневая сборка RAG пайплайна
# ==============================================================================


@dataclass
class RAGPipelineConfig:
    data: Any = field(default_factory=dict)
    model: Any = field(default_factory=dict)
    training: Any = field(default_factory=dict)
    training: Any = field(default_factory=dict)
    optimizer: Any = field(default_factory=dict)
    scheduler: Any = field(default_factory=dict)
    loss: Any = field(default_factory=dict)
    inference: Any = field(default_factory=dict)
    retrieval: Any = field(default_factory=dict)
    indexing: Any = field(default_factory=dict)
    api: FastAPIConfig = field(default_factory=FastAPIConfig)
