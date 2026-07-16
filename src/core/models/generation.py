"""
Модуль для текстовой генерации (Inference).
Оборачивает логику декодирования, сэмплирования и очистки токенов.
"""

import torch
from typing import List, Dict, Any, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import logging

logger = logging.getLogger(__name__)

class HFTextGenerator:
    """
    Индустриальный генератор текста для CausalLM моделей.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        generation_kwargs: Dict[str, Any]
    ):
        """
        Args:
            model: Загруженная модель (базовая или с LoRA).
            tokenizer: Токенизатор с padding_side="left" (Важно для генерации!).
            generation_kwargs: Параметры (temperature, max_new_tokens и т.д.) из конфига.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        
        # Переводим модель в режим инференса
        self.model.eval()

    @torch.inference_mode() # Отключаем градиенты для ускорения и экономии памяти
    def generate(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Генерирует ответы для одного текста или батча текстов.
        """
        if isinstance(texts, str):
            texts = [texts]

        # 1. Токенизация (обязательно на устройство модели)
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)

        # 2. Вызов встроенного метода генерации
        # Распаковываем **kwargs из конфига Гидры
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 3. Очистка ответа от входного промпта
        # model.generate возвращает [input_ids + generated_ids]
        # Нам нужно отрезать длину входных токенов, чтобы вернуть только ответ
        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_length:]

        # 4. Декодирование обратно в текст
        decoded_texts = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return decoded_texts