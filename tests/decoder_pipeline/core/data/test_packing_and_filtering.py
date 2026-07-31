from datasets import Dataset
from src.decoder_pipeline.core.data.transforms.packing import SequencePackingTransform

class TestSequencePackingTransform:
    def test_packs_short_sequences(self):
        ds = Dataset.from_dict({
            "input_ids": [[1, 2], [3, 4, 5]],
            "attention_mask": [[1, 1], [1, 1, 1]]
        })
        # При drop_remainder=False остаток [5] сохраняется отдельной строкой
        transform = SequencePackingTransform(packing_chunk_size=4, drop_remainder=False, num_proc=1)
        result = transform(ds)
        assert len(result["input_ids"]) == 2
        assert result["input_ids"][0] == [1, 2, 3, 4]
        assert result["input_ids"][1] == [5]