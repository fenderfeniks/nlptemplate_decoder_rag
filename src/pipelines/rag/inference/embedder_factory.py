"""Фабрика RAGInferenceEmbedder.

Логика фильтрации конфига через сигнатуру __init__ дублировалась в infer.py
и index_db.py. Вынесена сюда, чтобы изменения в RAGInferenceEmbedder
автоматически подхватывались везде.
"""

import inspect
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder


logger = logging.getLogger(__name__)

# Ключи, которые живут в inference.yaml для infer.py, но не являются
# параметрами RAGInferenceEmbedder.__init__() — фильтруем их один раз здесь.
_EMBEDDER_VALID_KEYS: frozenset[str] = frozenset(
    inspect.signature(RAGInferenceEmbedder.__init__).parameters
) | {"_target_"}


def build_embedder(
    cfg: DictConfig,
    base_model,
    pooler,
    tokenizer,
) -> RAGInferenceEmbedder:
    """Собрать RAGInferenceEmbedder, отфильтровав лишние ключи конфига.

    Args:
        cfg: Корневой конфиг. Читает cfg.rag_pipeline.inference.
        base_model: Уже собранный энкодер (с LoRA или без).
        pooler: Пулер эмбеддингов.
        tokenizer: Токенизатор.

    Returns:
        Готовый RAGInferenceEmbedder.
    """
    inference_cfg = cfg.rag_pipeline.inference
    embedder_cfg = OmegaConf.masked_copy(
        inference_cfg,
        [k for k in inference_cfg if k in _EMBEDDER_VALID_KEYS],
    )
    return hydra.utils.instantiate(
        embedder_cfg,
        model=base_model,
        pooler=pooler,
        tokenizer=tokenizer,
    )
