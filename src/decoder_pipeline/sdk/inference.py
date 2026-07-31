import asyncio
import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class LLMGenerationClient:
    """Легковесный клиент для асинхронного общения с LLM-сервером (vLLM / TGI)."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "my-decoder-model",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        logger.info("Инициализация клиента LLM (подключение к %s)", api_base)
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _generate_one(self, prompt: str) -> dict[str, str]:
        """Генерация для одного промпта."""
        response = await self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return {"prompt": prompt, "generated_text": response.choices[0].text}

    async def __call__(
        self,
        texts: str | list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[dict[str, str]]:
        """Параллельная батч-генерация через asyncio.gather."""
        if isinstance(texts, str):
            texts = [texts]

        # Позволяем переопределить параметры на уровне вызова
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
        return [
            {"prompt": prompt, "generated_text": resp.choices[0].text}
            for prompt, resp in zip(texts, responses)
        ]

    async def generate_stream(
        self,
        text: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Асинхронный стриминг токенов напрямую из vLLM."""
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
