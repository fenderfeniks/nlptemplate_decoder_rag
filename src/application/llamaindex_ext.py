# src/application/llamaindex_ext.py
"""LlamaIndex-обёртки над нашими пайплайнами.

Позволяет использовать LlamaIndex как альтернативный оркестратор поверх
тех же моделей: ``DecoderPipelineLLM`` — наш генератор как LLM-провайдер,
``RAGPipelineEmbedding`` — наш энкодер как embedding-провайдер.
"""

from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import PrivateAttr

from src.decoder_pipeline.sdk.inference import LLMGenerationPipeline
from src.rag_pipeline.inference.embedder import RAGInferenceEmbedder


class DecoderPipelineLLM(CustomLLM):
    """Обёртка над LLMGenerationPipeline для интеграции в LlamaIndex.

    LlamaIndex использует Pydantic под капотом, поэтому Python-объекты
    без Pydantic-схемы хранятся через ``PrivateAttr``.
    """

    _generator: LLMGenerationPipeline = PrivateAttr()

    def __init__(self, generator: LLMGenerationPipeline, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._generator = generator

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=1024,
            model_name="decoder_pipeline_model",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Синхронная генерация полного ответа."""
        results = self._generator([prompt])
        return CompletionResponse(text=results[0]["generated_text"])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Стриминговая генерация токенов.

        LlamaIndex требует синхронный генератор для этого метода.
        Наш генератор реализует асинхронный стриминг — используйте
        ``astream_complete`` для async-контекста.
        """
        # Не оборачиваем в вложенную функцию: raise внутри def gen() без yield
        # не сделает gen() генератором, и LlamaIndex получит исключение ещё
        # до первой итерации при вызове next().
        raise NotImplementedError(
            "Синхронный стриминг не поддерживается LLMGenerationPipeline. "
            "Используйте astream_complete() для async-контекста."
        )


class RAGPipelineEmbedding(BaseEmbedding):
    """Обёртка над RAGInferenceEmbedder для интеграции в LlamaIndex."""

    _embedder: RAGInferenceEmbedder = PrivateAttr()

    def __init__(self, embedder: RAGInferenceEmbedder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._embedder = embedder

    def _get_query_embedding(self, query: str) -> list[float]:
        """Векторизует один поисковый запрос."""
        vectors = self._embedder.encode([query])
        return vectors[0].tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async-версия: делегирует в синхронный метод (encode CPU-bound)."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Векторизует один документ при индексации."""
        vectors = self._embedder.encode([text])
        return vectors[0].tolist()

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async-версия для индексации документов."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Батчевая векторизация документов при индексации."""
        vectors = self._embedder.encode(texts, batch_size=32)
        return vectors.tolist()
