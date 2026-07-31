# tests/decoder_pipeline/core/data/test_collators.py
from unittest.mock import MagicMock
import torch

from src.decoder_pipeline.core.data.collators import InstructionDataCollator


class TestInstructionDataCollator:
    def test_masks_prompt_correctly(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.pad.return_value = {
            "input_ids": torch.tensor([[10, 20, 30, 40]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }

        collator = InstructionDataCollator(tokenizer=tokenizer, mask_prompt=True)
        features = [
            {"input_ids": [10, 20, 30, 40], "attention_mask": [1, 1, 1, 1], "prompt_len": 2}
        ]

        batch = collator(features)

        assert "labels" in batch
        # Первые 2 токена (промпт) должны быть замаскированы в -100
        assert batch["labels"][0, 0].item() == -100
        assert batch["labels"][0, 1].item() == -100
        # Ответ не замаскирован
        assert batch["labels"][0, 2].item() == 30
        assert batch["labels"][0, 3].item() == 40