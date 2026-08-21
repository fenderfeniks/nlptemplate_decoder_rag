# src/pipelines/decoder/inference/inference.py
import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pybreaker
from openai import AsyncOpenAI
from opentelemetry import trace

from src.api_gateway.resilience import CircuitBreakerError, llm_breaker

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class StreamChunk:
    """Фрагмент стримингового ответа с опциональными данными о токенах.

    Атрибуты:
        text:              Текстовый фрагмент (может быть пустым в последнем чанке).
        prompt_tokens:     Кол-во токенов промпта — заполняется только в последнем
                           чанке при ``stream_options={"include_usage": True}``.
        completion_tokens: Кол-во токенов ответа — аналогично.
        is_final:          True для последнего чанка (usage chunk от сервера).
    """
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    is_final: bool = False


class LLMGenerationClient:
    """HTTP-клиент для асинхронного общения с LLM-сервером (vLLM / llama.cpp).

    Оба сервера поднимают OpenAI-совместимый API, поэтому клиент один.
    Разница только в LLM_API_URL из env:
        llama.cpp:  http://localhost:8080/v1
        vLLM:       http://vllm-service:8000/v1

    Circuit breaker:
        generate_stream обёрнут в llm_breaker из resilience.py.
        При 5 ошибках подряд (default) breaker открывается и последующие
        вызовы fast-fail'ятся с CircuitBreakerError без ожидания таймаута.
        Это немедленно разгружает event loop вместо накопления зависших запросов.

    OTEL:
        Каждый вызов generate_stream создаёт span "llm.generate_stream"
        с атрибутами model, request_id (если передан через kwargs),
        статусом и количеством токенов из usage-чанка.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_concurrent_requests: int = 20,
        include_usage: bool = True,
    ) -> None:
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.include_usage = include_usage

        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        logger.info(
            "LLMGenerationClient инициализирован (api_base=%s, model=%s, include_usage=%s)",
            api_base, model_name, include_usage,
        )

    async def generate(self, prompt: str, **kwargs) -> object:
        """Полная генерация — возвращает полный ответ после завершения.

        Обёрнут в llm_breaker: при открытом breaker'е бросает CircuitBreakerError.
        """
        gen_kwargs = {
            "model": kwargs.pop("model", self.model_name),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        gen_kwargs.update(kwargs)

        async with self._semaphore:
            response = await llm_breaker.call_async(
                self.client.completions.create,
                prompt=prompt,
                **gen_kwargs,
            )
        return response

    async def generate_stream(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Потоковая генерация с circuit breaker'ом и OTEL span'ом.

        Circuit breaker:
            Только установка соединения (create) обёрнута в breaker —
            итерация чанков идёт уже вне него, иначе семафор держался бы
            на всё время стрима. Ошибки при итерации не засчитываются в breaker
            (соединение уже было успешным). Это осознанный trade-off:
            breaker защищает от недоступного сервера, а не от обрывов стрима.

        Args:
            prompt:     Промпт для генерации.
            request_id: UUID запроса для span-атрибута (опционально).
            **kwargs:   Переопределения параметров генерации для этого вызова.

        Yields:
            StreamChunk с текстом и (в финальном чанке) токен-статистикой.

        Raises:
            CircuitBreakerError: если llm_breaker открыт (fast fail).
            openai.APIError и его подклассы: при ошибках API.
        """
        gen_kwargs: dict = {
            "model": kwargs.pop("model", self.model_name),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
            "stream": True,
        }
        if self.include_usage:
            gen_kwargs["stream_options"] = {"include_usage": True}
        gen_kwargs.update(kwargs)

        with tracer.start_as_current_span("llm.generate_stream") as span:
            span.set_attribute("llm.model", self.model_name)
            span.set_attribute("llm.max_tokens", gen_kwargs["max_tokens"])
            span.set_attribute("llm.temperature", gen_kwargs["temperature"])
            if request_id:
                span.set_attribute("request.id", request_id)

            try:
                # Только connect обёрнут в breaker и семафор.
                # Семафор отпускается сразу после получения response-объекта —
                # до начала итерации чанков.
                async with self._semaphore:
                    response = await llm_breaker.call_async(
                        self.client.completions.create,
                        prompt=prompt,
                        **gen_kwargs,
                    )

            except pybreaker.CircuitBreakerError:
                span.set_attribute("llm.circuit_breaker", "open")
                span.set_status(trace.StatusCode.ERROR, "circuit breaker open")
                logger.warning(
                    "[%s] LLM circuit breaker OPEN — fast fail",
                    request_id or "no-id",
                )
                # Пробрасываем дальше — chat.py поймает и вернёт клиенту понятную ошибку
                raise

            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise

            # Итерация чанков — вне breaker'а и семафора
            prompt_tokens = 0
            completion_tokens = 0

            async for chunk in response:
                text = chunk.choices[0].text if chunk.choices else ""

                if not chunk.choices and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    yield StreamChunk(
                        text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        is_final=True,
                    )
                    return

                if text:
                    yield StreamChunk(text=text)

            # Записываем токены в span после завершения стрима
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.completion_tokens", completion_tokens)