import logging
from typing import Any

import httpx
from fastapi import HTTPException

from src.pipelines.decoder.core.prompts.manager import PromptManager
from src.pipelines.decoder.inference.inference import LLMGenerationClient


logger = logging.getLogger(__name__)


class RAGOrchestrator:
    def __init__(
        self,
        rag_api_url: str,
        llm_client: LLMGenerationClient,
        prompt_manager: PromptManager,
        default_template: str = "rag_qa_generation",
        default_top_k: int = 5,
        max_history_msgs: int = 10,
        http_timeout: float = 10.0,
    ) -> None:
        self.rag_api_url = rag_api_url.rstrip("/")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.default_template = default_template
        self.default_top_k = default_top_k
        self.max_history_msgs = max_history_msgs
        self.http_client = httpx.AsyncClient(timeout=http_timeout)

    async def build_prompt(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        template: str | None = None,
    ) -> str:

        # Эффект забывания переехал на использование атрибута класса
        if chat_history and len(chat_history) > self.max_history_msgs:
            logger.debug(
                "История диалога усечена с %d до %d сообщений",
                len(chat_history),
                self.max_history_msgs,
            )
            chat_history = chat_history[-self.max_history_msgs :]

        try:
            response = await self.http_client.post(
                f"{self.rag_api_url}/api/v1/search",
                json={"query": query, "top_k": top_k or self.default_top_k, "filters": filters},
            )
            response.raise_for_status()
            data = response.json()
            docs: list[dict[str, Any]] = data.get("results", [])

        except httpx.HTTPError as e:
            logger.error("Ошибка сети при обращении к RAG API: %s", e)
            raise HTTPException(status_code=502, detail="RAG Service is unavailable") from e

        if not docs:
            logger.warning("RAG API не вернул документов по запросу: '%s'", query[:80])

        context_str = "\n\n".join(
            f"[Документ {i}]: {doc.get('metadata', {}).get('text', '')}"
            for i, doc in enumerate(docs, 1)
        )

        return self.prompt_manager.render(
            template_name=template or self.default_template,
            question=query,
            context=context_str,
            chat_history=chat_history,
        )

    async def close(self) -> None:
        await self.http_client.aclose()
