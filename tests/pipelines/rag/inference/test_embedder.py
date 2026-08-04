# tests/pipelines/rag/inference/test_embedder.py
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder


class MockBatchEncoding(dict):
    """Эмулирует поведение HF BatchEncoding для переноса тензоров на девайс."""

    def to(self, device):
        return self


@pytest.fixture
def dummy_models():
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    pooler = MagicMock()
    pooler.to.return_value = pooler
    pooler.eval.return_value = pooler
    pooler.return_value = torch.ones(2, 4)

    tokenizer = MagicMock()
    tokenizer.return_value = MockBatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
    )

    return model, pooler, tokenizer


class TestRAGInferenceEmbedder:
    def test_invalid_precision(self, dummy_models):
        """Ошибка при инициализации с неизвестным precision."""
        model, pooler, tokenizer = dummy_models
        with pytest.raises(ValueError, match="Недопустимое значение precision"):
            RAGInferenceEmbedder(model, pooler, tokenizer, precision="fp64")

    def test_cpu_fallback_to_fp32(self, dummy_models):
        """Для CPU режим bf16 должен откатываться к fp32 и отключать autocast."""
        model, pooler, tokenizer = dummy_models
        embedder = RAGInferenceEmbedder(model, pooler, tokenizer, device="cpu", precision="bf16")

        assert embedder.dtype == torch.float32
        assert embedder._use_autocast is False

    def test_cuda_precision_mapping(self, dummy_models):
        """Для CUDA должны корректно маппиться типы и включаться autocast."""
        model, pooler, tokenizer = dummy_models
        embedder = RAGInferenceEmbedder(model, pooler, tokenizer, device="cuda", precision="fp16")

        assert embedder.dtype == torch.float16
        assert embedder._use_autocast is True
        assert embedder._autocast_device == "cuda"

    def test_encode_single_string(self, dummy_models):
        """Метод encode должен конвертировать одиночную строку в список и возвращать numpy array."""
        model, pooler, tokenizer = dummy_models
        embedder = RAGInferenceEmbedder(model, pooler, tokenizer, device="cpu", precision="fp32")

        result = embedder.encode("одна строка", batch_size=2)

        tokenizer.assert_called_once_with(
            ["одна строка"], padding="longest", truncation=True, max_length=512, return_tensors="pt"
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
