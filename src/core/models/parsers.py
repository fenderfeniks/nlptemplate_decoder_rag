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
        """
        if not raw_text:
            return ""

        text = raw_text

        # 1. Срезаем эхо промпта
        if self.strip_prompt and prompt and text.startswith(prompt):
            text = text[len(prompt):]

        # 2. Удаляем блоки заголовков Llama (например, <|start_header_id|>assistant<|end_header_id|>)
        # Используем re.DOTALL, чтобы захватить всё, что внутри, включая переносы строк
        text = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", text, flags=re.DOTALL)

        # 3. Удаляем оставшиеся системные токены
        if self.remove_special_tokens:
            special_tokens_pattern = r"<\|.*?\|>|</s>|<s>"
            text = re.sub(special_tokens_pattern, "", text)

        # 4. Удаляем дублирующиеся пробелы и пустые строки на краях
        if self.remove_extra_spaces:
            text = re.sub(r" +", " ", text) 
            text = text.strip()  # <--- Вот это удалит лишний \n, который остался после заголовка

        # 5. Обрезаем последнее предложение
        if self.trim_incomplete_sentence and text:
            last_punctuation = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_punctuation != -1 and last_punctuation < len(text) - 1:
                text = text[:last_punctuation + 1]

        return text