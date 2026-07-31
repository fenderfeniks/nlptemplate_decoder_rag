# src/schemas/system.py
from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class PathsConfig:
    root_dir: str = "."
    data_dir: str = "${paths.root_dir}/data"
    hf_cache_dir: str = "${paths.data_dir}/cache/hf_models"
    processed_data_dir: str = "${paths.data_dir}/processed/${pipeline_name}"
    log_dir: str = "${paths.root_dir}/logs/${pipeline_name}"
    mlflow_bd_dir: str = "${paths.root_dir}/logs"
    output_dir: str = "${paths.root_dir}/outputs/${pipeline_name}"
    model_dir: str = "${paths.root_dir}/models/${pipeline_name}"
    db_dir: str = "${paths.root_dir}/vector_db"


@dataclass
class SystemConfig:
    num_proc: int = 4
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EnvironmentConfig:
    name: str = "prod"


@dataclass
class MLFlowRegistryConfig:
    model_name: str = MISSING  # Будет браться из активного пайплайна
    register_on_success: bool = True
    artifact_path: str = "lora_weights"
    promote_to_staging: bool = True


@dataclass
class MLFlowLoggerConfig:
    _target_: str = "lightning.pytorch.loggers.MLFlowLogger"
    experiment_name: str = "nlp_project"
    tracking_uri: str = "sqlite:///${paths.log_dir}/mlflow.db"
    run_name: str = "${now:%Y-%m-%d_%H-%M-%S}"
    log_model: bool = False
    tags: dict[str, str] = field(default_factory=lambda: {"env": "${environment.name}"})


@dataclass
class RootLoggerConfig:
    pylightning: MLFlowLoggerConfig = field(default_factory=MLFlowLoggerConfig)
    registry: MLFlowRegistryConfig = field(default_factory=MLFlowRegistryConfig)


@dataclass
class PromptsConfig:
    rag_qa: str = ""
    summarization: str = ""
    translation: str = ""
    telegram_welcome: str = ""


@dataclass
class StringsConfig:
    bot: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class HydraConfig:
    run: dict[str, str] = field(
        default_factory=lambda: {"dir": "${paths.log_dir}/hydra/${now:%Y-%m-%d_%H-%M-%S}"}
    )
    job: dict[str, bool] = field(default_factory=lambda: {"chdir": True})


@dataclass
class VectorDBConfig:
    _target_: str = "src.utils.vector_db.FAISSVectorDB"
    embedding_dim: int = 1024
    index_type: str = "flat"
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    normalize_embeddings: bool = True
    insert_batch_size: int = 10000
    filter_fetch_multiplier: int = 5
    filter_max_fetch_multiplier: int = 50
