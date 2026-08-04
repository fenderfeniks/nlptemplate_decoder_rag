import pytest
from unittest.mock import patch, MagicMock

from src.pipelines.base.core.models.tokenization import HFTokenizerBuilder


class TestHFTokenizerBuilder:
    def test_invalid_padding_side(self):
        """Проверка исключения при невалидном padding_side."""
        with pytest.raises(ValueError, match="Недопустимое значение padding_side"):
            HFTokenizerBuilder(tokenizer_name="test", padding_side="center")

    @patch("src.pipelines.base.core.models.tokenization.AutoTokenizer.from_pretrained")
    def test_build_standard_tokenizer(self, mock_from_pretrained):
        """Проверка загрузки токенизатора со стандартными параметрами."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_from_pretrained.return_value = mock_tokenizer

        builder = HFTokenizerBuilder(
            tokenizer_name="fake-model",
            use_fast=True,
            padding_side="left",
            add_eos_token=True,
            trust_remote_code=False,
            cache_dir="/tmp/cache"
        )
        result = builder.build()

        mock_from_pretrained.assert_called_once_with(
            "fake-model",
            use_fast=True,
            add_eos_token=True,
            trust_remote_code=False,
            cache_dir="/tmp/cache"
        )
        assert result.padding_side == "left"
        assert result is mock_tokenizer

    @patch("src.pipelines.base.core.models.tokenization.AutoTokenizer.from_pretrained")
    def test_build_missing_pad_token(self, mock_from_pretrained):
        """Проверка фикса, когда pad_token отсутствует (заменяется на eos_token)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 99
        mock_from_pretrained.return_value = mock_tokenizer

        builder = HFTokenizerBuilder(tokenizer_name="fake-model")
        builder.build()

        assert mock_tokenizer.pad_token == "<eos>"
        assert mock_tokenizer.pad_token_id == 99

    @patch("src.pipelines.base.core.models.tokenization.AutoTokenizer.from_pretrained")
    def test_build_with_chat_template(self, mock_from_pretrained):
        """Проверка установки кастомного chat_template."""
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        builder = HFTokenizerBuilder(tokenizer_name="fake", chat_template="{% for message in messages %}")
        builder.build()

        assert mock_tokenizer.chat_template == "{% for message in messages %}"