import torch
import torch.nn as nn

class MultiTaskBERT(nn.Module):
    def __init__(self, base_builder, num_sentiment_classes: int, num_category_classes: int):
        super().__init__()
        # 1. Загружаем голое "туловище" (без голов) через наш билдер!
        self.encoder = base_builder.build() 
        
        # Узнаем размерность (например, 768 для BERT)
        hidden_size = self.encoder.config.hidden_size 
        
        # 2. Создаем свои независимые "головы"
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)
        self.category_head = nn.Linear(hidden_size, num_category_classes)

    def forward(self, input_ids, attention_mask):
        # Прогоняем текст через энкодер
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Берем вектор [CLS] токена
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        
        # Прогоняем через разные головы
        sentiment_logits = self.sentiment_head(cls_embedding)
        category_logits = self.category_head(cls_embedding)
        
        return {"sentiment": sentiment_logits, "category": category_logits}