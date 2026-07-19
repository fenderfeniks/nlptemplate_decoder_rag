import torch
import torch.nn as nn

class MultiTaskBERT(nn.Module):
    def __init__(self, base_builder, num_sentiment_classes: int, num_category_classes: int):
        super().__init__()
        self.encoder = base_builder.build() 
        hidden_size = self.encoder.config.hidden_size 
        
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)
        self.category_head = nn.Linear(hidden_size, num_category_classes)
        
        # Добавляем функции потерь внутрь модели для инкапсуляции
        self.loss_fn_sentiment = nn.CrossEntropyLoss()
        self.loss_fn_category = nn.CrossEntropyLoss()

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        sentiment_labels=None, 
        category_labels=None,
        **kwargs
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        
        sentiment_logits = self.sentiment_head(cls_embedding)
        category_logits = self.category_head(cls_embedding)
        
        loss = None
        # Если переданы лейблы (на этапе обучения/валидации), считаем комбинированный лосс
        if sentiment_labels is not None and category_labels is not None:
            loss_sent = self.loss_fn_sentiment(sentiment_logits, sentiment_labels)
            loss_cat = self.loss_fn_category(category_logits, category_labels)
            loss = loss_sent + loss_cat # Суммируем ошибки
            
        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits, 
            "category_logits": category_logits
        }