# src/application/orchestrator.py
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
import pybreaker
from opentelemetry import trace

from src.api_gateway.metrics import RAG_FALLBACK_TOTAL
from src.api_gateway.resilience import rag_breaker
from src.pipelines.decoder.core.prompts.manager import PromptManager
from src.pipelines.decoder.inference.inference import LLMGenerationClient

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class BuildPromptResult:
    """Результат build_prompt: промпт + метаданные для логирования и трейсинга.

    Разделяем данные (prompt) и observability (docs, rag_degraded) явно,
    чтобы chat.py мог передать docs в rag_logger без повторного вызова RAG.
    """
    prompt: str
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    rag_degraded: bool = False  # True если RAG был недоступен


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
    ) -> BuildPromptResult:
        """Строит промпт: ретривал из RAG + рендер шаблона.

        Возвращает BuildPromptResult вместо голой строки, чтобы chat.py
        получил retrieved_docs для rag_logger без повторного запроса к RAG.

        Graceful degradation:
            При недоступности RAG (сеть, circuit breaker) продолжаем работу
            с пустым контекстом вместо HTTP 502. LLM отвечает на основе
            своих знаний — хуже, но лучше полного отказа сервиса.
            Факт деградации фиксируется в метрике, логе и заголовке ответа.

        Returns:
            BuildPromptResult с промптом, документами и флагом деградации.
        """
        if chat_history and len(chat_history) > self.max_history_msgs:
            logger.debug(
                "История диалога усечена с %d до %d сообщений",
                len(chat_history),
                self.max_history_msgs,
            )
            chat_history = chat_history[-self.max_history_msgs:]

        docs, rag_degraded = await self._retrieve_with_fallback(
            query=query,
            top_k=top_k or self.default_top_k,
            filters=filters,
        )

        context_str = "\n\n".join(
            f"[Документ {i}]: {doc.get('metadata', {}).get('text', '')}"
            for i, doc in enumerate(docs, 1)
        )

        prompt = self.prompt_manager.render(
            template_name=template or self.default_template,
            question=query,
            context=context_str,
            chat_history=chat_history,
        )

        return BuildPromptResult(
            prompt=prompt,
            retrieved_docs=docs,
            rag_degraded=rag_degraded,
        )

    async def _retrieve_with_fallback(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Вызывает RAG API с circuit breaker'ом. При ошибке деградирует.

        Returns:
            (docs, rag_degraded): docs пустой список при деградации.
        """
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.top_k", top_k)

            try:
                docs = await rag_breaker.call_async(
                    self._call_rag_api,
                    query=query,
                    top_k=top_k,
                    filters=filters,
                )
                span.set_attribute("rag.doc_count", len(docs))
                span.set_attribute("rag.degraded", False)

                if not docs:
                    logger.warning("RAG API не вернул документов по запросу: '%s'", query[:80])

                return docs, False

            except pybreaker.CircuitBreakerError:
                # Breaker открыт — fast fail без ожидания таймаута
                logger.warning(
                    "RAG circuit breaker OPEN — деградируем до LLM-only (query='%s')",
                    query[:80],
                )
                span.set_attribute("rag.degraded", True)
                span.set_attribute("rag.degraded_reason", "circuit_breaker_open")
                RAG_FALLBACK_TOTAL.labels(reason="circuit_breaker").inc()
                return [], True

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                # Сеть недоступна — записываем ошибку в breaker и деградируем
                logger.warning(
                    "RAG API недоступен (%s) — деградируем до LLM-only",
                    type(e).__name__,
                )
                span.set_attribute("rag.degraded", True)
                span.set_attribute("rag.degraded_reason", type(e).__name__)
                RAG_FALLBACK_TOTAL.labels(reason="network_error").inc()
                return [], True

            except httpx.HTTPStatusError as e:
                # 5xx от RAG API — тоже деградируем
                logger.warning(
                    "RAG API вернул %d — деградируем до LLM-only",
                    e.response.status_code,
                )
                span.set_attribute("rag.degraded", True)
                span.set_attribute("rag.degraded_reason", f"http_{e.response.status_code}")
                RAG_FALLBACK_TOTAL.labels(reason=f"http_{e.response.status_code}").inc()
                return [], True

    async def _call_rag_api(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Сырой HTTP-вызов к RAG API. Обёрнут в rag_breaker в caller'е.

        Выбрасывает httpx-исключения — они считаются ошибками для breaker'а.
        """
        response = await self.http_client.post(
            f"{self.rag_api_url}/api/v1/search",
            json={"query": query, "top_k": top_k, "filters": filters},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    async def close(self) -> None:
        await self.http_client.aclose()