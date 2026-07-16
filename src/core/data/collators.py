import torch
from typing import Any
from transformers import PreTrainedTokenizerBase

class DynamicTextCollator:
    """
    Коллатор для сборки батчей, токенизации и динамического паддинга.
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int = 512,
        text_column: str = "text",
        target_column: str = "label",
        is_causal_lm: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.target_column = target_column
        self.is_causal_lm = is_causal_lm

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [feature[self.text_column] for feature in features]
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if self.is_causal_lm:
            labels = batch["input_ids"].clone()
            
            # БЕЗОПАСНАЯ МАСКИРОВКА (Защита от бага pad_token == eos_token)
            # Если attention_mask == 0, значит это паддинг. Туда ставим -100.
            labels[batch["attention_mask"] == 0] = -100
            
            batch["labels"] = labels
            
        elif self.target_column in features[0]:
            targets = [feature[self.target_column] for feature in features]
            batch["labels"] = torch.tensor(targets, dtype=torch.long)
            
        return batch
  
    
class TripletTextCollator:
    """Коллатор для подготовки батчей под Triplet Loss (Anchor, Positive, Negative)."""
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int = 512,
        anchor_column: str = "anchor",
        positive_column: str = "positive",
        negative_column: str = "negative"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.anchor_col = anchor_column
        self.pos_col = positive_column
        self.neg_col = negative_column

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Вспомогательная функция для токенизации списка текстов
        def _tokenize(texts):
            return self.tokenizer(
                texts, padding=True, truncation=True, 
                max_length=self.max_length, return_tensors="pt"
            )

        # Собираем три отдельных батча
        anchor_batch = _tokenize([f[self.anchor_col] for f in features])
        pos_batch = _tokenize([f[self.pos_col] for f in features])
        neg_batch = _tokenize([f[self.neg_col] for f in features])

        # Отдаем словарь тензоров (модель будет ждать именно эти ключи в forward)
        return {
            "anchor_input_ids": anchor_batch["input_ids"],
            "anchor_attention_mask": anchor_batch["attention_mask"],
            "pos_input_ids": pos_batch["input_ids"],
            "pos_attention_mask": pos_batch["attention_mask"],
            "neg_input_ids": neg_batch["input_ids"],
            "neg_attention_mask": neg_batch["attention_mask"],
        }