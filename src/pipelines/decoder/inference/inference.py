# src/pipelines/decoder/inference/inference.py
import asyncio
import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class LLMGenerationClient:
    """Легковесный клиент для асинхронного общения с LLM-сервером (vLLM / TGI).

    Использует OpenAI-совместимый API — работает с любым сервером,
    поддерживающим ``/v1/completions`` (vLLM, TGI, llama.cpp server).
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "my-decoder-model",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        """
        Args:
            api_base: Базовый URL сервера (без trailing slash).
            api_key: API-ключ — для локальных серверов обычно ``'EMPTY'``.
            model_name: Имя модели как оно зарегистрировано на сервере.
            temperature: Температура сэмплирования по умолчанию.
            max_tokens: Максимальное число генерируемых токенов по умолчанию.
        """
        logger.info("Инициализация клиента LLM (подключение к %s)", api_base)
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
        self,
        texts: str | list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Параллельная батч-генерация через ``asyncio.gather``.

        Все запросы из батча отправляются одновременно — latency определяется
        самым медленным из них, а не суммой.

        Args:
            texts: Один промпт или список промптов.
            max_tokens: Переопределяет ``self.max_tokens`` для этого вызова.
            temperature: Переопределяет ``self.temperature`` для этого вызова.

        Returns:
            Список сгенерированных строк в том же порядке что и входные тексты.
        """
        if isinstance(texts, str):
            texts = [texts]

        _max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        _temperature = temperature if temperature is not None else self.temperature

        tasks = [
            self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=_max_tokens,
                temperature=_temperature,
            )
            for prompt in texts
        ]
        responses = await asyncio.gather(*tasks)
        return [resp.choices[0].text for resp in responses]

    async def generate_stream(
        self,
        text: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Асинхронный стриминг токенов напрямую из vLLM.

        Args:
            text: Одиночный промпт (стриминг не поддерживает батчи).
            max_tokens: Переопределяет ``self.max_tokens`` для этого вызова.
            temperature: Переопределяет ``self.temperature`` для этого вызова.

        Yields:
            Строковые фрагменты по мере генерации.
        """
        response = await self.client.completions.create(
            model=self.model_name,
            prompt=text,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text
