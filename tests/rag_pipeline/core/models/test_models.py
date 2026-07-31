# tests/rag_pipeline/core/test_models.py
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.rag_pipeline.core.models.pooling import Pooler
from src.rag_pipeline.core.models.tokenization import HFTokenizerBuilder


class TestPooler:
    @pytest.fixture
    def token_embeddings(self):
        # Батч 2, 4 токена, скрытость 3. Значения - счетчик для предсказуемости
        emb = torch.tensor([
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0]],
            [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ])
        return emb

    @pytest.fixture
    def attention_mask(self):
        # Первый пример: 3 токена + 1 паддинг. Второй: 2 токена + 2 паддинга
        return torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0]
        ])

    def test_cls_pooling(self, token_embeddings, attention_mask):
        """Проверяет режим 'cls' (берется 0-й токен)."""
        pooler = Pooler(pooling_mode="cls", normalize=False)
        res = pooler(token_embeddings, attention_mask)
        
        assert res.shape == (2, 3)
        assert torch.allclose(res[0], torch.tensor([1.0, 1.0, 1.0]))
        assert torch.allclose(res[1], torch.tensor([4.0, 4.0, 4.0]))

    def test_mean_pooling(self, token_embeddings, attention_mask):
        """Проверяет режим 'mean' с учетом маски."""
        pooler = Pooler(pooling_mode="mean", normalize=False)
        res = pooler(token_embeddings, attention_mask)
        
        # Среднее для [1,2,3] = 2. Для [4,5] = 4.5
        assert torch.allclose(res[0], torch.tensor([2.0, 2.0, 2.0]))
        assert torch.allclose(res[1], torch.tensor([4.5, 4.5, 4.5]))

    def test_last_token_pooling(self, token_embeddings, attention_mask):
        """Проверяет режим 'last_token'."""
        pooler = Pooler(pooling_mode="last_token", normalize=False)
        res = pooler(token_embeddings, attention_mask)
        
        # Последний не-pad токен: индекс 2 для 1-го, индекс 1 для 2-го
        assert torch.allclose(res[0], torch.tensor([3.0, 3.0, 3.0]))
        assert torch.allclose(res[1], torch.tensor([5.0, 5.0, 5.0]))

    def test_normalization_applied(self, token_embeddings, attention_mask):
        """Нормализация должна приводить длину вектора к 1."""
        pooler = Pooler(pooling_mode="cls", normalize=True)
        res = pooler(token_embeddings, attention_mask)
        
        norms = torch.linalg.norm(res, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            Pooler(pooling_mode="unknown_mode")


class TestHFTokenizerBuilder:
    @patch("src.rag_pipeline.core.models.tokenization.AutoTokenizer.from_pretrained")
    def test_adds_pad_token_if_missing(self, mock_from_pretrained):
        """Токенизатор без pad_token должен получить eos_token как pad."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 1
        mock_from_pretrained.return_value = mock_tokenizer

        builder = HFTokenizerBuilder("test-model")
        tokenizer = builder.build()

        assert tokenizer.pad_token == "<eos>"
        assert tokenizer.pad_token_id == 1

    @patch("src.rag_pipeline.core.models.tokenization.AutoTokenizer.from_pretrained")
    def test_overrides_padding_side(self, mock_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        builder = HFTokenizerBuilder("test-model", padding_side="left")
        tokenizer = builder.build()

        assert tokenizer.padding_side == "left"

    def test_invalid_padding_side_raises(self):
        with pytest.raises(ValueError):
            HFTokenizerBuilder("test", padding_side="center")