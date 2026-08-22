# src/pipelines/decoder/inference/response_cleaner.py
"""Постпроцессинг сырого вывода LLM.

Объединяет очистку спецтокенов, эхо-промпта, Markdown-артефактов
и обрезку незавершённых предложений в один настраиваемый класс.

Фабричные методы:
    ResponseCleaner.for_stream()  — безопасная конфигурация для почанковой очистки
                                    в стриминге (trim и markdown отключены).
    ResponseCleaner.for_batch()   — полная очистка для небатчевой / post-stream очистки.
"""

import logging
import re


logger = logging.getLogger(__name__)

# Последний знак конца предложения с последующим пробелом или концом строки.
# Ищем последнее вхождение в тексте — обрезаем после него.
# Использует lookahead (?=\s|$) чтобы не захватывать пробел в результат.
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

# Markdown-блоки кода: ```lang\n...\n``` — в любом месте текста, включая середину.
# re.DOTALL нужен чтобы . матчило переносы строк внутри блока.
_MARKDOWN_BLOCK_RE = re.compile(r"```[a-zA-Z]*\n.*?```", re.DOTALL)

# Llama-заголовки: <|start_header_id|>assistant<|end_header_id|>
_LLAMA_HEADER_RE = re.compile(r"<\|start_header_id\|>.*?<\|end_header_id\|>", re.DOTALL)

# Прочие системные токены: <|eot_id|>, <|im_end|>, </s>, <s> и т.п.
_SPECIAL_TOKENS_RE = re.compile(r"<\|.*?\|>|</s>|<s>")

# Множественные пробелы (не переносы строк) -> один пробел
_EXTRA_SPACES_RE = re.compile(r" +")


class ResponseCleaner:
    """Универсальный постпроцессор ответов генеративной модели.

    Порядок шагов:
        1. Срезаем эхо-промпт (если модель вернула его в начале ответа).
        2. Удаляем Llama-заголовки ``<|start_header_id|>...<|end_header_id|>``.
        3. Удаляем прочие системные токены (``<|...|>``, ``</s>``, ``<s>``).
        4. Удаляем Markdown-блоки кода (`` ```lang ... ``` ``).
        5. Нормализуем пробелы и обрезаем края.
        6. Обрезаем незавершённое последнее предложение.

    Для стриминга используй фабричный метод ``for_stream()`` —
    он отключает шаги 4 и 6, небезопасные для пофрагментной обработки:
    markdown-блок может быть разбит по границам чанков, а trim_incomplete_sentence
    обрежет середину потока.
    """

    def __init__(
        self,
        strip_prompt: bool = True,
        remove_special_tokens: bool = True,
        remove_markdown_blocks: bool = True,
        remove_extra_spaces: bool = True,
        trim_incomplete_sentence: bool = True,
    ) -> None:
        """
        Args:
            strip_prompt: Удалять ли исходный промпт из начала ответа.
            remove_special_tokens: Удалять ли системные токены вроде ``<s>``, ``</s>``.
            remove_markdown_blocks: Удалять ли блоки кода Markdown (в любом месте текста).
                Небезопасно для пофрагментной очистки стрима — блок может быть разбит
                на несколько чанков.
            remove_extra_spaces: Схлопывать ли множественные пробелы в один.
            trim_incomplete_sentence: Обрезать ли текст после последнего завершённого
                предложения (оканчивается на ``.``, ``!`` или ``?``).
                Использует lookahead ``(?=\\s|$)`` — не срабатывает на точках внутри
                аббревиатур и числах (``2.3``, ``т.е.``) если за ними нет пробела/конца.
                Небезопасно для пофрагментной очистки стрима — обрежет середину потока.
        """
        self.strip_prompt = strip_prompt
        self.remove_special_tokens = remove_special_tokens
        self.remove_markdown_blocks = remove_markdown_blocks
        self.remove_extra_spaces = remove_extra_spaces
        self.trim_incomplete_sentence = trim_incomplete_sentence

    # ------------------------------------------------------------------
    # Фабричные методы
    # ------------------------------------------------------------------

    @classmethod
    def for_stream(cls) -> "ResponseCleaner":
        """Конфигурация для пофрагментной очистки в стриминге.

        Отключает:
            - ``strip_prompt``: completions API промпт не эхоит.
            - ``remove_markdown_blocks``: блок может быть разбит по чанкам.
            - ``trim_incomplete_sentence``: обрежет середину потока.

        Оставляет:
            - ``remove_special_tokens``: <|eot_id|>, </s> могут прилететь
              в последнем чанке даже при ``skip_special_tokens=True`` на стороне сервера.
            - ``remove_extra_spaces``: безопасно на уровне одного чанка.
        """
        return cls(
            strip_prompt=False,
            remove_special_tokens=True,
            remove_markdown_blocks=False,
            remove_extra_spaces=True,
            trim_incomplete_sentence=False,
        )

    @classmethod
    def for_batch(cls) -> "ResponseCleaner":
        """Полная очистка для батча или полного текста после стрима.

        Включает все шаги включая trim_incomplete_sentence.
        Используется в HFTextGenerator и для постпроцессинга
        ``generated_text`` в метриках стримингового эндпоинта.
        """
        return cls(
            strip_prompt=True,
            remove_special_tokens=True,
            remove_markdown_blocks=True,
            remove_extra_spaces=True,
            trim_incomplete_sentence=True,
        )

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def clean(self, raw_text: str, prompt: str | None = None) -> str:
        """Очищает сгенерированный текст.

        Args:
            raw_text: Сырой текст, сгенерированный моделью.
            prompt: Исходный промпт — необходим если ``strip_prompt=True``.

        Returns:
            Очищенная строка. Пустая строка если ``raw_text`` пустой.
        """
        if not raw_text:
            return ""

        text = raw_text

        # 1. Срезаем эхо-промпт
        if self.strip_prompt and prompt and text.startswith(prompt):
            text = text[len(prompt) :]

        # 2. Llama-заголовки
        text = _LLAMA_HEADER_RE.sub("", text)

        # 3. Прочие системные токены
        if self.remove_special_tokens:
            text = _SPECIAL_TOKENS_RE.sub("", text)

        # 4. Markdown-блоки кода — ищем в любом месте текста, не только в начале/конце
        if self.remove_markdown_blocks:
            text = _MARKDOWN_BLOCK_RE.sub("", text)

        # 5. Нормализация пробелов (только горизонтальные — переносы строк сохраняем)
        if self.remove_extra_spaces:
            text = _EXTRA_SPACES_RE.sub(" ", text)
            text = text.strip()

        # 6. Обрезка незавершённого последнего предложения.
        # Ищем все вхождения знаков конца предложения и берём позицию последнего.
        # Lookahead (?=\s|$) защищает от срабатывания на "2.3" или "т.е." —
        # после точки в числе/аббревиатуре нет пробела перед следующим символом.
        if self.trim_incomplete_sentence and text:
            matches = list(_SENTENCE_END_RE.finditer(text))
            if matches:
                last_end = matches[-1].end()
                # Обрезаем только если после последнего знака есть незавершённый хвост
                if last_end < len(text):
                    text = text[:last_end]

        return text
