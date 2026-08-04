from unittest.mock import MagicMock, patch

import pytest
import torch

from src.pipelines.decoder.inference.generator import HFTextGenerator


class MockBatchEncoding(dict):
    """Эмулирует поведение HF BatchEncoding для переноса тензоров на девайс."""

    def to(self, device):
        return self


@pytest.fixture
def dummy_components():
    """Моки для токенизатора и модели Hugging Face."""
    tokenizer = MagicMock()
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.model_max_length = 512

    # Используем MockBatchEncoding вместо обычного словаря
    tokenizer.return_value = MockBatchEncoding(
        {"input_ids": torch.tensor([[10, 20]]), "attention_mask": torch.tensor([[1, 1]])}
    )
    tokenizer.batch_decode.return_value = ["raw text 1", "raw text 2"]

    model = MagicMock()
    param = MagicMock()
    param.device = "cpu"
    model.parameters.return_value = iter([param])
    model.generate.return_value = torch.tensor([[10, 20, 30, 40]])

    return model, tokenizer


class TestHFTextGenerator:
    def test_init_fixes_padding_side(self, dummy_components):
        """Проверка принудительной установки padding_side='left' для батч-генерации."""
        model, tokenizer = dummy_components
        HFTextGenerator(model, tokenizer, generation_kwargs={})
        assert tokenizer.padding_side == "left"

    def test_merge_kwargs(self, dummy_components):
        """Проверка переопределения базовых аргументов."""
        model, tokenizer = dummy_components
        gen = HFTextGenerator(model, tokenizer, generation_kwargs={"temperature": 0.7, "top_k": 50})

        merged = gen._merge_kwargs({"temperature": 0.9, "max_new_tokens": 100})
        assert merged == {"temperature": 0.9, "top_k": 50, "max_new_tokens": 100}

    def test_generate_batch(self, dummy_components):
        """Проверка полного пайплайна генерации батча (со слайсингом токенов)."""
        model, tokenizer = dummy_components
        gen = HFTextGenerator(model, tokenizer, generation_kwargs={})

        # Мокаем клинер, чтобы он просто возвращал то, что получил
        gen.cleaner.clean = MagicMock(side_effect=lambda raw_text, prompt: f"Cleaned: {raw_text}")

        result = gen.generate(["промпт 1", "промпт 2"], max_new_tokens=10)

        assert len(result) == 2
        # Обновляем ожидаемые значения под новую фикстуру
        assert result == ["Cleaned: raw text 1", "Cleaned: raw text 2"]

        # Проверяем, что в batch_decode передались только сгенерированные токены (без промпта)
        call_tensor = tokenizer.batch_decode.call_args[0][0]
        assert call_tensor.shape == (1, 2)
        assert call_tensor[0].tolist() == [30, 40]

    def test_generate_stream_invalid_input(self, dummy_components):
        """Стриминг должен падать при попытке передать список промптов."""
        model, tokenizer = dummy_components
        gen = HFTextGenerator(model, tokenizer, generation_kwargs={})

        with pytest.raises(ValueError, match="поддерживает только одиночные строки"):
            list(gen.generate_stream(["prompt1", "prompt2"]))

    @patch("src.pipelines.decoder.inference.generator.Thread")
    @patch("src.pipelines.decoder.inference.generator.TextIteratorStreamer")
    def test_generate_stream_success(self, mock_streamer_cls, mock_thread, dummy_components):
        """Проверка потоковой генерации с моком треда и стримера."""
        model, tokenizer = dummy_components
        gen = HFTextGenerator(model, tokenizer, generation_kwargs={})

        # Настраиваем фейковый стример
        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = [
            "chunk 1",
            "",
            " chunk 2",
        ]  # Пустая строка должна быть отфильтрована
        mock_streamer_cls.return_value = mock_streamer

        # Собираем результат генератора
        result = list(gen.generate_stream("тестовый промпт"))

        assert result == ["chunk 1", " chunk 2"]
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
