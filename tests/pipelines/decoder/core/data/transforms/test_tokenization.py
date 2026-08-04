# tests/pipelines/decoder/core/data/transforms/test_tokenization.py
import pytest
from datasets import Dataset

from src.pipelines.decoder.core.data.transforms.tokenization import TokenizationTransform


class DummyTokenizer:
    """Сериализуемый класс для замены MagicMock в многопроцессорном map."""
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Простая эмуляция: каждый символ = 1 токен
        return {
            "input_ids": [[1] * len(t) for t in texts],
            "attention_mask": [[1] * len(t) for t in texts]
        }
    
    def apply_chat_template(self, messages, **kwargs):
        return {
            "input_ids": [[1, 2, 3]], 
            "attention_mask": [[1, 1, 1]]
        }


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()


class TestTokenizationTransform:
    def test_invalid_init(self, dummy_tokenizer):
        with pytest.raises(ValueError, match="Неизвестный режим токенизации"):
            TokenizationTransform(tokenizer=dummy_tokenizer, mode="invalid")
        with pytest.raises(ValueError, match="max_length должен быть положительным"):
            TokenizationTransform(tokenizer=dummy_tokenizer, max_length=0)

    def test_missing_required_column_skipped(self, dummy_tokenizer):
        """Проверка пропуска, если нет основной колонки для выбранного режима."""
        ds = Dataset.from_dict({"wrong_column": ["text"]})
        transform = TokenizationTransform(tokenizer=dummy_tokenizer, mode="cpt")
        result = transform(ds)
        assert result is ds

    def test_sft_missing_target_column_skipped(self, dummy_tokenizer):
        """Проверка пропуска, если в режиме sft есть prompt, но нет target."""
        ds = Dataset.from_dict({"prompt": ["Текст промпта"]})
        transform = TokenizationTransform(tokenizer=dummy_tokenizer, mode="sft")
        result = transform(ds)
        assert result is ds

    def test_cpt_tokenization(self, dummy_tokenizer):
        """Проверка CPT: текстовая колонка удаляется, остаются тензоры."""
        ds = Dataset.from_dict({"text": ["abc", "defg"]})
        transform = TokenizationTransform(tokenizer=dummy_tokenizer, mode="cpt", num_proc=1)
        
        result = transform(ds)
        assert "text" not in result.column_names
        assert result["input_ids"] == [[1, 1, 1], [1, 1, 1, 1]]

    def test_sft_tokenization_with_prompt_len(self, dummy_tokenizer):
        """Проверка SFT: separator добавляется, вычисляется prompt_len."""
        ds = Dataset.from_dict({"prompt": ["Q1"], "target": ["A1"]})
        transform = TokenizationTransform(
            tokenizer=dummy_tokenizer, mode="sft", separator=" SEP ", num_proc=1
        )
        
        result = transform(ds)
        assert "prompt" not in result.column_names
        # Длина Q1 + SEP + A1 = 2 + 5 + 2 = 9
        assert len(result["input_ids"][0]) == 9
        # Длина промпта: Q1 + SEP = 2 + 5 = 7
        assert result["prompt_len"][0] == 7

    def test_chat_tokenization(self, dummy_tokenizer):
        """Проверка режима chat через apply_chat_template."""
        ds = Dataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})
        transform = TokenizationTransform(tokenizer=dummy_tokenizer, mode="chat", num_proc=1)
        
        result = transform(ds)
        assert "messages" not in result.column_names
        assert result["input_ids"] == [[1, 2, 3]]