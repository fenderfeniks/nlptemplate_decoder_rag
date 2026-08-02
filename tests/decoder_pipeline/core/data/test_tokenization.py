# tests/decoder_pipeline/core/data/test_tokenization.py
from datasets import Dataset
from src.decoder_pipeline.core.data.transforms.tokenization import TokenizationTransform

class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, list):
            return {
                "input_ids": [[10, 20, 30, 40]] * len(texts), 
                "attention_mask": [[1, 1, 1, 1]] * len(texts)
            }
        return {"input_ids": [10, 20], "attention_mask": [1, 1]}
"""
class TestTokenizationTransform:
    def test_tokenizes_sft_prompt_target(self):
        ds = Dataset.from_dict({
            "prompt": ["Вопрос?"],
            "response": ["Ответ!"]
        })

        transform = TokenizationTransform(
            tokenizer=DummyTokenizer(),
            prompt_column="prompt",
            target_column="response",
            num_proc=1
        )
        result = transform(ds)

        assert "input_ids" in result.column_names
        assert "attention_mask" in result.column_names
        assert "prompt_len" in result.column_names"""