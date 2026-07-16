"""
Модуль постпроцессинга и очистки ответов LLM.
Заменяет ручную зачистку строк и делает выход модели безопасным и чистым.
"""

import re
import logging

logger = logging.getLogger(__name__)

class ResponseCleaner:
    def __init__(
        self,
        strip_prompt: bool = True,
        remove_special_tokens: bool = True,
        remove_extra_spaces: bool = True,
        trim_incomplete_sentence: bool = True
    ):
        self.strip_prompt = strip_prompt
        self.remove_special_tokens = remove_special_tokens
        self.remove_extra_spaces = remove_extra_spaces
        self.trim_incomplete_sentence = trim_incomplete_sentence

    def clean(self, raw_text: str, prompt: str = None) -> str:
        """
        Основной метод очистки текста.
        
        Args:
            raw_text (str): Сырой ответ от LLM.
            prompt (str, optional): Исходный промпт для срезания эха.
        """
        if not raw_text:
            return ""

        text = raw_text

        # 1. Срезаем эхо промпта
        if self.strip_prompt and prompt and text.startswith(prompt):
            text = text[len(prompt):]

        # 2. Удаляем системные токены (например, <|im_start|>, <|eot_id|> и т.д.)
        if self.remove_special_tokens:
            # Регулярка для поиска любых конструкций вида <|...|> или </s>
            special_tokens_pattern = r"<\|.*?\|>|</s>|<s>"
            text = re.sub(special_tokens_pattern, "", text)

        # 3. Удаляем дублирующиеся пробелы и пустые строки на краях
        if self.remove_extra_spaces:
            text = re.sub(r" +", " ", text)  # Склеиваем множественные пробелы
            text = text.strip()

        # 4. Обрезаем последнее предложение, если оно не завершено
        if self.trim_incomplete_sentence and text:
            # Ищем последние знаки препинания, завершающие мысль
            last_punctuation = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            # Если нашли знак препинания и после него идет незаконченный кусок текста
            if last_punctuation != -1 and last_punctuation < len(text) - 1:
                # Обрезаем текст ровно по последний завершенный знак препинания
                text = text[:last_punctuation + 1]

        return text