import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class LLMGenerationClient:
    """Легковесный клиент для асинхронного общения с LLM-сервером (vLLM / TGI)."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",  # URL вашего vLLM контейнера
        api_key: str = "EMPTY",  # vLLM по умолчанию принимает любой ключ
        model_name: str = "my-decoder-model",
    ) -> None:
        logger.info("Инициализация клиента LLM (Подключение к %s)", api_base)
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

    async def __call__(
        self, texts: str | list[str], max_tokens: int = 1024
    ) -> list[dict[str, str]]:
        """Асинхронная батч-генерация через API."""
        if isinstance(texts, str):
            texts = [texts]

        results = []
        # В реальном проде здесь можно использовать asyncio.gather для параллельных запросов
        for prompt in texts:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            results.append({"prompt": prompt, "generated_text": response.choices[0].text})
        return results

    async def generate_stream(self, text: str, max_tokens: int = 1024) -> AsyncGenerator[str, None]:
        """Асинхронный стриминг токенов напрямую из vLLM."""
        response = await self.client.completions.create(
            model=self.model_name,
            prompt=text,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text
