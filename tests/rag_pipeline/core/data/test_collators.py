# tests/rag_pipeline/core/data/test_collators.py
from unittest.mock import MagicMock

import pytest
import torch

from src.rag_pipeline.core.data.collators import ContrastiveDataCollator, IndexingDataCollator


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    # Имитация работы padding: просто преобразует списки в тензоры 
    tok.pad.side_effect = lambda data, **kwargs: {
        "input_ids": torch.tensor(data["input_ids"]),
        "attention_mask": torch.tensor(data["attention_mask"]),
    }
    return tok


class TestIndexingDataCollator:
    def test_collates_and_preserves_meta(self, mock_tokenizer):
        collator = IndexingDataCollator(tokenizer=mock_tokenizer, text_column="text")
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "text": "A", "metadata": {"id": 1}},
            {"input_ids": [3, 4], "attention_mask": [1, 1], "text": "B", "metadata": {"id": 2}},
        ]
        
        batch = collator(features)
        
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["text"] == ["A", "B"]
        assert batch["metadata"] == [{"id": 1}, {"id": 2}]


class TestContrastiveDataCollator:
    def test_collates_triplets(self, mock_tokenizer):
        collator = ContrastiveDataCollator(tokenizer=mock_tokenizer)
        features = [
            {
                "query_input_ids": [1], "query_attention_mask": [1],
                "pos_input_ids": [2], "pos_attention_mask": [1],
                "neg_input_ids": [3], "neg_attention_mask": [1],
            }
        ]
        
        batch = collator(features)
        
        assert "query_input_ids" in batch
        assert "pos_input_ids" in batch
        assert "neg_input_ids" in batch
        assert isinstance(batch["pos_input_ids"], torch.Tensor)