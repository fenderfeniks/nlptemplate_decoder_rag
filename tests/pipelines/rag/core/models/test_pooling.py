import pytest
import torch
from src.pipelines.rag.core.models.pooling import Pooler

class TestPooler:
    def test_invalid_mode(self):
        """Проверка исключения при неизвестном режиме пулинга."""
        with pytest.raises(ValueError, match="Неизвестный режим пулинга"):
            Pooler(pooling_mode="invalid")

    def test_cls_pooling(self):
        """Проверка пулинга по первому токену (CLS)."""
        pooler = Pooler(pooling_mode="cls", normalize=False)
        # [batch_size=2, seq_len=3, hidden_size=4]
        token_embs = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        mask = torch.ones(2, 3)
        
        res = pooler(token_embs, mask)
        # Должен вернуться нулевой индекс seq_len
        assert torch.allclose(res, token_embs[:, 0, :])

    def test_mean_pooling(self):
        """Проверка усреднения с учетом attention_mask."""
        pooler = Pooler(pooling_mode="mean", normalize=False)
        token_embs = torch.ones(2, 3, 4)
        
        # Первая последовательность: 2 токена. Вторая: 1 токен.
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        res = pooler(token_embs, mask)
        
        # Поскольку исходный тензор - единицы, усреднение единиц даст единицы,
        # если маскирование и деление на длину работает правильно.
        assert torch.allclose(res, torch.ones(2, 4))

    def test_last_token_pooling(self):
        """Проверка пулинга по последнему реальному токену."""
        pooler = Pooler(pooling_mode="last_token", normalize=False)
        token_embs = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        
        mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
        res = pooler(token_embs, mask)
        
        # Для 0-го батча последний токен имеет индекс 2
        assert torch.allclose(res[0], token_embs[0, 2, :])
        # Для 1-го батча последний токен имеет индекс 0
        assert torch.allclose(res[1], token_embs[1, 0, :])

    def test_normalization(self):
        """Проверка L2-нормализации векторов."""
        pooler = Pooler(pooling_mode="cls", normalize=True)
        token_embs = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3)
        
        res = pooler(token_embs, mask)
        norms = torch.linalg.norm(res, dim=1)
        
        # Норма каждого вектора должна быть равна 1.0
        assert torch.allclose(norms, torch.ones_like(norms))