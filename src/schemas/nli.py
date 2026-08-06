# src/schemas/nli.py
from dataclasses import dataclass, field
from typing import Any


# ==============================================================================
# Архитектура
# ==============================================================================


@dataclass
class NLIArchitectureConfig:
    model_name_or_path: str = "cross-encoder/nli-roberta-base"
    mlflow_model_name: str = "nli-roberta-base"
    auto_model_class: str = "transformers.AutoModelForSequenceClassification"
    torch_dtype: str = "float32"
    attn_implementation: str | None = None
    trust_remote_code: bool = False


@dataclass
class NLIModelConfig:
    architecture: NLIArchitectureConfig = field(default_factory=NLIArchitectureConfig)
    builder: Any = field(default_factory=dict)


# ==============================================================================
# Корневая сборка NLI пайплайна
# ==============================================================================


@dataclass
class NLIPipelineConfig:
    """Минимальный пайплайн-конфиг для NLI-модели.

    Не содержит data/training/optimizer — NLI-модель не обучается,
    только скачивается через prepare_artifacts и используется как judge.
    """

    model: NLIModelConfig = field(default_factory=NLIModelConfig)
