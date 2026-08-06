# src/schemas/main.py
from dataclasses import dataclass, field
from typing import Any

from src.schemas.application import ApplicationConfig, TgBotConfig
from src.schemas.decoder import DecoderPipelineConfig
from src.schemas.evaluation import EvaluationConfig  # новое
from src.schemas.nli import NLIPipelineConfig  # новое
from src.schemas.rag import RAGPipelineConfig
from src.schemas.system import (
    EnvironmentConfig,
    HydraConfig,
    PathsConfig,
    PromptsConfig,
    RootLoggerConfig,
    StorageRouterConfig,
    StringsConfig,
    SystemConfig,
    VectorDBConfig,
)


@dataclass
class ConfigSchema:
    """Корневая схема конфигурации всего фреймворка."""

    seed: int = 42
    max_length: int = 2048
    project_name: str = "industrial_nlp_template"
    pipeline_name: str = "decoder_pipeline"
    resume_training: bool = False
    ckpt_path: str | None = None
    metrics_output_path: str = "${paths.output_dir}metrics.json"
    drift_threshold: float | None = None
    drift_metric_key: str = "test_perplexity"
    text: str | None = None
    manifest_uri: str | None = None
    incremental: bool = False
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logger: RootLoggerConfig = field(default_factory=RootLoggerConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    strings: StringsConfig = field(default_factory=StringsConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)
    manifest: Any | None = None
    storage: Any | None = None
    storage_router: StorageRouterConfig = field(default_factory=StorageRouterConfig)
    vector_db: VectorDBConfig | None = None
    decoder_pipeline: DecoderPipelineConfig | None = None
    rag_pipeline: RAGPipelineConfig | None = None
    nli_pipeline: NLIPipelineConfig | None = None  # новое
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)  # новое
    # Application layer
    tg_bot: TgBotConfig = field(default_factory=TgBotConfig)
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
