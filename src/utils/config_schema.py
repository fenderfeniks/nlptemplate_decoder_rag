# src/core/utils/config_schema.py
from dataclasses import dataclass
from typing import Any

@dataclass
class PathsConfig:
    log_dir: str
    data_dir: str
    hf_cache_dir: str

@dataclass
class ConfigSchema:
    """Главная схема, описывающая весь наш main.yaml"""
    paths: PathsConfig
    model: Any       # Оставляем Any для гибкости вложенных блоков
    data: Any
    trainer: Any
    seed: int
    project_name: str