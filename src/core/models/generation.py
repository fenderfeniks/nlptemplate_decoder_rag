"""
Модуль для текстовой генерации (Inference).
Оборачивает логику декодирования, сэмплирования и очистки токенов.
"""

import torch
import logging
from typing import List, Dict, Any, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from hydra.utils import instantiate

# Импортируем наш постпроцессор
from src.core.models.parsers import ResponseCleaner

logger = logging.getLogger(__name__)

class HFTextGenerator:
    """
    Индустриальный генератор текста для CausalLM моделей.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        generation_kwargs: Dict[str, Any],
        cleaner_cfg: Any = None
    ):
        """
        Args:
            model: Загруженная модель (базовая или с LoRA).
            tokenizer: Токенизатор с padding_side="left" (Важно для генерации!).
            generation_kwargs: Параметры (temperature, max_new_tokens и т.д.) из конфига.
            cleaner_cfg: Конфиг Гидры для инициализации ResponseCleaner.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        
        # Инстанцируем очиститель из конфига Гидры. Если его нет — берем дефолтный.
        self.cleaner = instantiate(cleaner_cfg) if cleaner_cfg else ResponseCleaner()
        
        # Переводим модель в режим инференса
        self.model.eval()

    @torch.inference_mode() # Отключаем градиенты для ускорения и экономии памяти
    # ИСПРАВЛЕНИЕ 2: Добавляем **kwargs, чтобы эндпоинты могли безопасно переопределять параметры (Race Condition fix)
    def generate(self, texts: Union[str, List[str]], **kwargs) -> List[str]:
        """
        Генерирует ответы для одного текста или батча текстов.
        """
        if isinstance(texts, str):
            texts = [texts]

        # ИСПРАВЛЕНИЕ 1: Надежно получаем device для inputs, даже если модель распределена по GPU
        device = next(self.model.parameters()).device

        # 1. Токенизация (обязательно на устройство модели)
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)

        # Мержим глобальные параметры (из конфига) с локальными (из API)
        # Это позволяет динамически менять max_new_tokens на один запрос, не мутируя self.generation_kwargs
        current_gen_kwargs = self.generation_kwargs.copy()
        current_gen_kwargs.update(kwargs)

        # 2. Вызов встроенного метода генерации
        # Распаковываем обновленные **kwargs
        generated_ids = self.model.generate(
            **inputs,
            **current_gen_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 3. Очистка ответа от входного промпта на уровне токенов
        # model.generate возвращает [input_ids + generated_ids]
        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_length:]

        # 4. Декодирование обратно в текст
        # Оставляем спецтокены, чтобы наш ResponseCleaner вырезал их наверняка
        decoded_texts = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )

        # 5. Финальный постпроцессинг (строковая очистка)
        final_responses = []
        for original_prompt, raw_response in zip(texts, decoded_texts):
            # Прогоняем через наш класс. Передаем prompt на случай, 
            # если токенизатор плохо обрезал эхо на шаге 3.
            cleaned_text = self.cleaner.clean(raw_text=raw_response, prompt=original_prompt)
            final_responses.append(cleaned_text)

        return final_responses