"""RAGOrchestrator — API Gateway для связывания микросервисов поиска и генерации."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from src.decoder_pipeline.core.prompts.manager import PromptManager
from src.decoder_pipeline.sdk.inference import LLMGenerationClient  # Наш новый легкий клиент


logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Связывает микросервисы поиска (RAG API) и генерации (LLM API).

    Принимает вопрос пользователя, делает HTTP-запрос за релевантными документами,
    формирует промпт и стримит ответ от LLM.
    """

    def __init__(
        self,
        rag_api_url: str,
        llm_client: LLMGenerationClient,
        prompt_manager: PromptManager,
        default_template: str = "rag_qa",
        default_top_k: int = 5,
    ) -> None:
        """
        Args:
            rag_api_url: Базовый URL микросервиса RAG (например, http://rag-api:8001).
            llm_client: Инстанс легкого клиента для LLM.
            prompt_manager: Менеджер шаблонов промптов.
            default_template: Имя шаблона по умолчанию.
            default_top_k: Количество документов для ретривала по умолчанию.
        """
        self.rag_api_url = rag_api_url.rstrip("/")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.default_template = default_template
        self.default_top_k = default_top_k
        self.http_client = httpx.AsyncClient(timeout=10.0)

    async def _retrieve_and_build_prompt(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        template: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Асинхронно запрашивает документы из RAG API и формирует промпт.

        Больше не нужен asyncio.to_thread, так как сетевой запрос нативно асинхронный!
        """
        try:
            # 1. Запрос к RAG API
            response = await self.http_client.post(
                f"{self.rag_api_url}/api/v1/search",
                json={"query": query, "top_k": top_k, "filters": filters},
            )
            response.raise_for_status()

            # Ожидаем, что RAG API вернет {"results": [...]}
            data = response.json()
            docs = data.get("results", [])

        except httpx.HTTPError as e:
            logger.error("Ошибка сети при обращении к RAG API: %s", e)
            raise HTTPException(status_code=502, detail="RAG Service is unavailable") from e

        # 2. Сборка контекста из полученных документов
        context_parts = [f"[Документ {i}]: {doc.get('text', '')}" for i, doc in enumerate(docs, 1)]
        context_str = "\n\n".join(context_parts)

        if not docs:
            logger.warning("RAG API не вернул документов по запросу: '%s'", query[:80])

        # 3. Рендер промпта через PromptManager
        prompt = self.prompt_manager.render(
            template_name=template,
            question=query,
            context=context_str,
        )
        return prompt, docs

    async def ask_stream(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        template: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Асинхронный стриминг ответа от LLM."""

        prompt, docs = await self._retrieve_and_build_prompt(
            query=query,
            top_k=top_k or self.default_top_k,
            filters=filters,
            template=template or self.default_template,
        )

        logger.info("ask_stream(): найдено %d документов, запуск стриминга...", len(docs))

        # Передаем готовый промпт в легкий LLM клиент
        async for chunk in self.llm_client.generate_stream(prompt):
            yield chunk

    async def close(self):
        """Важно для корректного закрытия соединений httpx."""
        await self.http_client.aclose()
