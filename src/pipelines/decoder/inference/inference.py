# src/pipelines/decoder/inference/inference.py
import asyncio
import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class LLMGenerationClient:
    """HTTP-клиент для асинхронного общения с LLM-сервером (vLLM / llama.cpp).

    Оба сервера поднимают OpenAI-совместимый API, поэтому клиент один.
    Разница только в LLM_API_URL из env:
        llama.cpp:  http://localhost:8080/v1
        vLLM:       http://vllm-service:8000/v1
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_concurrent_requests: int = 20,
    ) -> None:
        # api_base сохраняем как атрибут — health.py читает его для connectivity check
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

        # Семафор создаём сразу — __init__ вызывается внутри lifespan (async контекст),
        # поэтому event loop уже запущен и семафор привязан к правильному loop.
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        logger.info("LLMGenerationClient инициализирован (api_base=%s, model=%s)", api_base, model_name)

    async def generate(self, prompt: str, **kwargs) -> object:
        """Полная генерация — возвращает полный ответ после завершения.

        Семафор держится только на время HTTP round-trip, не на чтение ответа.
        """
        gen_kwargs = {
            "model": kwargs.pop("model", self.model_name),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        gen_kwargs.update(kwargs)

        async with self._semaphore:
            response = await self.client.completions.create(
                prompt=prompt,
                **gen_kwargs,
            )

        # Возвращаем сырой объект OpenAI — response_adapter разбирает его.
        return response

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        gen_kwargs = {
            "model": kwargs.pop("model", self.model_name),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
            "stream": True,
        }
        gen_kwargs.update(kwargs)

        async with self._semaphore:
            response = await self.client.completions.create(
                prompt=prompt,
                **gen_kwargs,
            )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text