from datasets import Dataset
from src.decoder_pipeline.core.data.transforms.validation import ValidationTransform

def dummy_pipeline(text): 
    return text

class TestValidationTransform:
    def test_filters_invalid_sft_records(self):
        ds = Dataset.from_dict({
            "prompt": ["Нормальный вопрос", "а"],
            "target": ["Нормальный ответ", "Короткий промпт отпадет"]
        })
        transform = ValidationTransform(pipeline=dummy_pipeline, num_proc=1)
        result = transform(ds)
        assert len(result) == 1
        assert result["prompt"][0] == "Нормальный вопрос"