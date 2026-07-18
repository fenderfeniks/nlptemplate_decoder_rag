# src/core/data/__init__.py

# Явно прописываем, что мы отдаем наружу из этой папки
from .cleaners import BaseCleaner, RegexCleaner, TextCleaningPipeline
from .collators import DynamicTextCollator
from .datasets import NLPDatasetAdapter
from .builder import NLPDataModule  # Исправлено с .datamodule на .builder

# Ограничиваем то, что импортируется при from src.core.data import *
__all__ = [
    "BaseCleaner",
    "RegexCleaner",
    "TextCleaningPipeline",
    "DynamicTextCollator",
    "NLPDatasetAdapter",
    "NLPDataModule"
]