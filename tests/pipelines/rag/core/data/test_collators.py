# tests/pipelines/rag/core/data/test_collators.py
import pytest
import torch
from unittest.mock import MagicMock

from src.pipelines.rag.core.data.collators import IndexingDataCollator, ContrastiveDataCollator


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    
    # Мок метода pad: просто оборачивает входящие списки в тензоры
    # Настоящий pad делает дополнение нулями, но для теста маршрутизации достаточно тензоров
    def pad_side_effect(features, **kwargs):
        return {
            "input_ids": torch.tensor(features["input_ids"]),
            "attention_mask": torch.tensor(features["attention_mask"])
        }
    tokenizer.pad.side_effect = pad_side_effect
    return tokenizer


class TestIndexingDataCollator:
    def test_basic_padding(self, mock_tokenizer):
        """Проверка паддинга базовых токенов (без метаданных и текстов)."""
        collator = IndexingDataCollator(tokenizer=mock_tokenizer)
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1]},
            {"input_ids": [3, 4], "attention_mask": [1, 1]}
        ]
        
        batch = collator(features)
        
        assert "input_ids" in batch
        assert "text" not in batch
        assert batch["input_ids"].shape == (2, 2)
        mock_tokenizer.pad.assert_called_once()

    def test_passing_text_and_metadata(self, mock_tokenizer):
        """Коллатор должен пробрасывать текстовые поля и метаданные as-is."""
        collator = IndexingDataCollator(tokenizer=mock_tokenizer, text_column="content")
        features = [
            {"input_ids": [1], "attention_mask": [1], "content": "text1", "metadata": {"id": 1}},
            {"input_ids": [2], "attention_mask": [1], "content": "text2", "metadata": {"id": 2}}
        ]
        
        batch = collator(features)
        
        assert batch["text"] == ["text1", "text2"]
        assert batch["metadata"] == [{"id": 1}, {"id": 2}]


class TestContrastiveDataCollator:
    def test_contrastive_mode(self, mock_tokenizer):
        """Батчинг запросов и позитивных документов (без негативных)."""
        collator = ContrastiveDataCollator(tokenizer=mock_tokenizer)
        features = [
            {
                "query_input_ids": [1], "query_attention_mask": [1],
                "pos_input_ids": [2], "pos_attention_mask": [1]
            }
        ]
        
        batch = collator(features)
        
        assert "query_input_ids" in batch
        assert "pos_input_ids" in batch
        assert "neg_input_ids" not in batch
        
        # Метод pad должен быть вызван дважды (для query и для pos)
        assert mock_tokenizer.pad.call_count == 2

    def test_triplet_mode(self, mock_tokenizer):
        """Батчинг с негативными документами."""
        collator = ContrastiveDataCollator(tokenizer=mock_tokenizer)
        features = [
            {
                "query_input_ids": [1], "query_attention_mask": [1],
                "pos_input_ids": [2], "pos_attention_mask": [1],
                "neg_input_ids": [3], "neg_attention_mask": [1]
            }
        ]
        
        batch = collator(features)
        
        assert "query_input_ids" in batch
        assert "pos_input_ids" in batch
        assert "neg_input_ids" in batch
        
        # Метод pad должен быть вызван трижды (query, pos, neg)
        assert mock_tokenizer.pad.call_count == 3
        assert batch["neg_input_ids"].shape == (1, 1)
        assert batch["neg_input_ids"][0].item() == 3