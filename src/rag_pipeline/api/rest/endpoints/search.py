# src/rag_pipeline/api/rest/endpoints/search.py
import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from src.rag_pipeline.api.metrics import RAG_SEARCH_REQUESTS_TOTAL, RAG_SEARCH_TIME
from src.rag_pipeline.api.rest.dependencies import get_retriever
from src.rag_pipeline.api.rest.limiter import limiter
from src.rag_pipeline.api.schemas import Document, SearchRequest, SearchResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


@router.post("/search", response_model=SearchResponse)
@limiter.limit("20/minute")
async def search_endpoint(request: Request, body: SearchRequest) -> SearchResponse:
    """Векторный поиск по базе знаний.

    Векторизует запрос через энкодер и ищет ближайшие документы в FAISS.
    Поиск выполняется в threadpool (``asyncio.to_thread``), чтобы не блокировать
    event loop во время CPU-bound инференса и FAISS-поиска.
    """
    retriever = get_retriever(request)
    RAG_SEARCH_REQUESTS_TOTAL.labels(source="rest").inc()

    logger.debug(
        "Search request: query='%s', top_k=%d, filters=%s",
        body.query[:80],
        body.top_k,
        body.filters,
    )

    try:
        t0 = time.perf_counter()
        with RAG_SEARCH_TIME.labels(source="rest").time():
            raw_results = await asyncio.to_thread(
                retriever.search,
                query=body.query,
                top_k=body.top_k,
                filter_metadata=body.filters,
            )
        query_time_ms = (time.perf_counter() - t0) * 1000

        documents = [Document(score=r["score"], metadata=r["metadata"]) for r in raw_results]
        return SearchResponse(
            results=documents,
            total=len(documents),
            query_time_ms=round(query_time_ms, 2),
        )

    except Exception:
        # exc_info=True добавляет полный traceback в лог — критично для диагностики
        logger.error("Ошибка векторного поиска (query='%s')", body.query[:80], exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при выполнении поиска.")
