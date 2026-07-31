# src/schemas/main.py
from dataclasses import dataclass, field

from src.schemas.application import ApplicationConfig, TgBotConfig
from src.schemas.decoder import DecoderPipelineConfig
from src.schemas.rag import RAGPipelineConfig
from src.schemas.system import (
    EnvironmentConfig,
    HydraConfig,
    PathsConfig,
    PromptsConfig,
    RootLoggerConfig,
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
    metrics_output_path: str = "metrics.json"
    drift_threshold: float | None = None
    drift_metric_key: str = "test_perplexity"
    text: str | None = None

    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logger: RootLoggerConfig = field(default_factory=RootLoggerConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    strings: StringsConfig = field(default_factory=StringsConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)

    vector_db: VectorDBConfig | None = None
    decoder_pipeline: DecoderPipelineConfig | None = None
    rag_pipeline: RAGPipelineConfig | None = None

    # Application layer
    tg_bot: TgBotConfig = field(default_factory=TgBotConfig)
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
