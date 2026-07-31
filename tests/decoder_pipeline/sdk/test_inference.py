# tests/decoder_pipeline/sdk/test_inference.py
from unittest.mock import MagicMock, patch

import pytest

from src.decoder_pipeline.inference.generator import HFTextGenerator
from src.decoder_pipeline.inference.response_cleaner import ResponseCleaner


class TestResponseCleaner:
    @pytest.fixture
    def cleaner(self):
        return ResponseCleaner()

    def test_strip_prompt(self, cleaner):
        """Удаляет промпт, если модель вернула его как эхо."""
        prompt = "Как дела?"
        raw = "Как дела? Хорошо."
        assert cleaner.clean(raw, prompt=prompt) == "Хорошо."

    def test_remove_markdown_blocks(self, cleaner):
        raw = "```python\nprint('Hello')\n```"
        assert cleaner.clean(raw) == "print('Hello')"

    def test_trim_incomplete_sentence(self, cleaner):
        """Обрезает предложение без точки на конце."""
        raw = "Первое предложение. Второе незаконченное"
        assert cleaner.clean(raw) == "Первое предложение."


class TestHFTextGenerator:
    @patch("src.decoder_pipeline.inference.generator.torch")
    def test_generator_overrides_padding_side(self, mock_torch):
        """Для батч-генерации padding_side обязан быть 'left'."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.padding_side = "right"

        mock_model = MagicMock()

        generator = HFTextGenerator(mock_model, mock_tokenizer, generation_kwargs={})
        assert generator.tokenizer.padding_side == "left"

    @patch("src.decoder_pipeline.inference.generator.torch")
    def test_generate_merges_kwargs(self, mock_torch):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_cleaner = MagicMock()
        mock_cleaner.clean.return_value = "Очищенный ответ"

        generator = HFTextGenerator(
            mock_model, mock_tokenizer, generation_kwargs={"temperature": 0.7}
        )
        generator.cleaner = mock_cleaner

        generator.generate("Вопрос", temperature=0.1, max_new_tokens=10)

        # Проверяем, что локальные kwargs перезаписали глобальные
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_new_tokens"] == 10
