from src.pipelines.decoder.inference.response_cleaner import ResponseCleaner


class TestResponseCleaner:
    def test_empty_string(self):
        """Проверка безопасной работы с пустой строкой."""
        cleaner = ResponseCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""

    def test_strip_prompt(self):
        """Проверка удаления эхо-промпта из начала ответа."""
        cleaner = ResponseCleaner(strip_prompt=True)
        raw = "Вопрос: как дела? Ответ: нормально."
        prompt = "Вопрос: как дела? "

        assert cleaner.clean(raw, prompt=prompt) == "Ответ: нормально."
        # Если промпт не совпадает с началом текста, текст не меняется
        assert cleaner.clean(raw, prompt="Другой вопрос") == raw

    def test_remove_llama_headers(self):
        """Проверка удаления Llama 3 заголовков."""
        cleaner = ResponseCleaner()
        raw = "<|start_header_id|>assistant<|end_header_id|>\nПривет! Как дела?"
        assert cleaner.clean(raw) == "Привет! Как дела?"

    def test_remove_special_tokens(self):
        """Проверка удаления системных токенов."""
        cleaner = ResponseCleaner(remove_special_tokens=True)
        raw = "Привет </s> <s> <|eot_id|> мир"
        assert cleaner.clean(raw) == "Привет мир"

    def test_remove_markdown_blocks(self):
        """Проверка удаления блоков кода."""
        cleaner = ResponseCleaner(remove_markdown_blocks=True)
        raw = "Вот код: ```python\nprint(1)\n``` и всё."
        assert cleaner.clean(raw) == "Вот код: и всё."

    def test_remove_extra_spaces(self):
        """Проверка нормализации пробелов и обрезки краев."""
        cleaner = ResponseCleaner(remove_extra_spaces=True)
        raw = "  Слишком    много   пробелов  "
        assert cleaner.clean(raw) == "Слишком много пробелов"

    def test_trim_incomplete_sentence(self):
        """Проверка отсечения незавершенных предложений."""
        cleaner = ResponseCleaner(trim_incomplete_sentence=True)

        # Обрезаем незавершенный хвост
        assert cleaner.clean("Это конец. А это не") == "Это конец."

        # Защита от срабатывания на числах и аббревиатурах (нет пробела после точки)
        assert (
            cleaner.clean("Цена 3.5 доллара. И это т.д. всё.")
            == "Цена 3.5 доллара. И это т.д. всё."
        )

        # Если вообще нет знаков конца предложения, строка остается как есть
        assert cleaner.clean("Текст без конца") == "Текст без конца"

    def test_full_pipeline(self):
        """Проверка работы всех шагов одновременно."""
        cleaner = ResponseCleaner()
        prompt = "User: hi\n"
        raw = "User: hi\n<|start_header_id|>assistant<|end_header_id|> Hello! Here is code: ```bash\nls\n``` </s> Done. Not fin"

        # Исправляем строку: текст перед блоком кода остается
        assert cleaner.clean(raw, prompt=prompt) == "Hello! Here is code: Done."
