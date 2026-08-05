from unittest.mock import AsyncMock, patch

import pytest

from src.pipelines.decoder.inference.inference import LLMGenerationClient


# Вспомогательные классы для эмуляции ответа OpenAI SDK
class MockChoice:
    def __init__(self, text):
        self.text = text
        self.choices = [self]  # Для стриминга: chunk.choices[0].text


class MockResponse:
    def __init__(self, text):
        self.choices = [MockChoice(text)]


async def mock_async_generator():
    """Эмулирует асинхронный стрим из OpenAI API."""
    yield MockResponse("hello")
    yield MockResponse(" world")
    # Эмулируем пустой чанк (должен быть проигнорирован)
    yield MockResponse("")


@pytest.mark.asyncio
class TestLLMGenerationClient:
    @patch("src.pipelines.decoder.inference.inference.AsyncOpenAI")
    async def test_call_single_prompt(self, mock_openai_cls):
        """Проверка батч-генерации (1 промпт)."""
        mock_client = mock_openai_cls.return_value
        mock_client.completions.create = AsyncMock(return_value=MockResponse("ответ 1"))

        client = LLMGenerationClient()
        result = await client.generate("один промпт", max_tokens=50)

        assert len(result) == 1
        assert result[0] == "ответ 1"
        mock_client.completions.create.assert_called_once_with(
            model="my-decoder-model", prompt="один промпт", max_tokens=50, temperature=0.7
        )

    @patch("src.pipelines.decoder.inference.inference.AsyncOpenAI")
    async def test_call_batch_prompts(self, mock_openai_cls):
        """Проверка параллельной батч-генерации через asyncio.gather."""
        mock_client = mock_openai_cls.return_value
        # Возвращаем разные ответы для разных вызовов
        mock_client.completions.create = AsyncMock(
            side_effect=[MockResponse("ответ 1"), MockResponse("ответ 2")]
        )

        client = LLMGenerationClient()
        result = await client.generate(["промпт 1", "промпт 2"], temperature=0.9)

        assert len(result) == 2
        assert result[0] == "ответ 1"
        assert result[1] == "ответ 2"
        assert mock_client.completions.create.call_count == 2

    @patch("src.pipelines.decoder.inference.inference.AsyncOpenAI")
    async def test_generate_stream(self, mock_openai_cls):
        """Проверка асинхронного генератора."""
        mock_client = mock_openai_cls.return_value

        mock_client.completions.create = AsyncMock(return_value=mock_async_generator())

        client = LLMGenerationClient()
        chunks = []

        async for chunk in client.generate_stream("тест", max_tokens=10):
            chunks.append(chunk)

        assert chunks == ["hello", " world"]
        mock_client.completions.create.assert_called_once_with(
            model="my-decoder-model", prompt="тест", max_tokens=10, temperature=0.7, stream=True
        )
