# tests/pipelines/decoder/core/data/test_collators.py
import pytest
import torch
from unittest.mock import MagicMock

from src.pipelines.decoder.core.data.collators import InstructionDataCollator


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    # Эмулируем поведение метода pad
    def pad_side_effect(features, **kwargs):
        # Дополняем нулями до самой длинной последовательности
        max_len = max(len(ids) for ids in features["input_ids"])
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(features["input_ids"], features["attention_mask"]):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_ids),
            "attention_mask": torch.tensor(padded_masks)
        }
    tokenizer.pad.side_effect = pad_side_effect
    return tokenizer


class TestInstructionDataCollator:
    def test_padding_and_pad_masking(self, mock_tokenizer):
        """Проверка динамического паддинга и маскирования токенов паддинга."""
        collator = InstructionDataCollator(tokenizer=mock_tokenizer, mask_prompt=False)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4], "attention_mask": [1]},
        ]
        
        batch = collator(features)
        
        # Длина должна быть 3 (по самой длинной)
        assert batch["input_ids"].shape == (2, 3)
        # Второй элемент должен быть дополнен нулями
        assert batch["input_ids"][1].tolist() == [4, 0, 0]
        # Паддинг в labels должен быть замаскирован (-100)
        assert batch["labels"][1].tolist() == [4, -100, -100]

    def test_masking_by_prompt_len(self, mock_tokenizer):
        """Проверка маскирования промпта по предварительно вычисленному prompt_len."""
        collator = InstructionDataCollator(tokenizer=mock_tokenizer, mask_prompt=True)
        features = [
            {"input_ids": [10, 20, 30, 40], "attention_mask": [1, 1, 1, 1], "prompt_len": 2},
        ]
        
        batch = collator(features)
        
        # Первые 2 токена должны стать -100
        assert batch["labels"][0].tolist() == [-100, -100, 30, 40]

    def test_masking_by_response_template(self, mock_tokenizer):
        """Проверка маскирования по поиску шаблона ответа (response_template)."""
        mock_tokenizer.encode.return_value = [99] # Шаблон ответа - токен 99
        
        collator = InstructionDataCollator(
            tokenizer=mock_tokenizer, 
            mask_prompt=True, 
            response_template="<Answer>"
        )
        
        features = [
            # Промпт (10, 20), Шаблон (99), Ответ (30)
            {"input_ids": [10, 20, 99, 30], "attention_mask": [1, 1, 1, 1]},
        ]
        
        batch = collator(features)
        
        # Замаскировано должно быть всё ДО и ВКЛЮЧАЯ шаблон (10, 20, 99 -> -100)
        assert batch["labels"][0].tolist() == [-100, -100, -100, 30]

    def test_truncation_to_max_sequence_length(self, mock_tokenizer):
        """Проверка жесткой обрезки (fallback truncation), если последовательность длиннее max_length."""
        collator = InstructionDataCollator(tokenizer=mock_tokenizer, max_sequence_length=2)
        features = [
            {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        ]
        
        batch = collator(features)
        assert batch["input_ids"].shape == (1, 2)
        assert batch["input_ids"][0].tolist() == [1, 2]