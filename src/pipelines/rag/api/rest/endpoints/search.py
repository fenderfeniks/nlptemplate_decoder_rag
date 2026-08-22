# src/pipelines/rag/api/rest/endpoints/search.py
import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from src.pipelines.rag.api.metrics import (
    RAG_EMPTY_RESULTS_TOTAL,
    RAG_ERRORS_TOTAL,
    RAG_REQUEST_DURATION_SECONDS,
    RAG_RESULTS_RETURNED,
    RAG_SEARCH_REQUESTS_TOTAL,
    RAG_TOP_SCORE,
)
from src.pipelines.rag.api.rest.dependencies import get_retriever
from src.pipelines.rag.api.rest.limiter import limiter
from src.pipelines.rag.api.schemas import Document, SearchRequest, SearchResponse
from src.pipelines.rag.inference.retriever import HybridRetriever


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Retrieval"])

_DEFAULT_LIMIT = "20/minute"


@router.post("/search", response_model=SearchResponse)
@limiter.limit(_DEFAULT_LIMIT)
async def search_endpoint(
    request: Request,
    body: SearchRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> SearchResponse:
    """Гибридный векторный поиск по базе знаний с опциональным реранкингом.

    Пайплайн:
        1. encode(query)            -> query_vector (CPU/GPU)
        2. vector_db.search_hybrid  -> top-N кандидатов (Qdrant: dense + BM25 + RRF)
        3. [опц.] reranker.rerank   -> пересортировка через Cross-Encoder

    Весь пайплайн выполняется в threadpool (``asyncio.to_thread``),
    чтобы не блокировать event loop во время CPU-bound инференса.
    """
    RAG_SEARCH_REQUESTS_TOTAL.labels(source="rest").inc()

    logger.debug(
        "Search request: query='%s', top_k=%d, filters=%s",
        body.query[:80],
        body.top_k,
        body.filters,
    )

    t0 = time.perf_counter()
    status = "success"

    try:
        raw_results: list[dict] = await asyncio.to_thread(
            retriever.search,
            query=body.query,
            top_k=body.top_k,
            filter_metadata=body.filters,
        )
    except Exception as err:
        status = "error"
        RAG_ERRORS_TOTAL.labels(source="rest", error_type="search_error").inc()
        logger.error("Ошибка поиска (query='%s')", body.query[:80], exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при выполнении поиска.") from err
    finally:
        elapsed = time.perf_counter() - t0
        RAG_REQUEST_DURATION_SECONDS.labels(source="rest", status=status).observe(elapsed)

    query_time_ms = elapsed * 1000

    # --- Метрики качества результатов ---
    n_results = len(raw_results)
    RAG_RESULTS_RETURNED.labels(source="rest").observe(n_results)

    if n_results == 0:
        RAG_EMPTY_RESULTS_TOTAL.labels(source="rest").inc()
        logger.warning("Поиск вернул 0 результатов (query='%s')", body.query[:80])
    else:
        # score первого документа: после реранкинга это cross_encoder_score,
        # иначе — косинусное сходство из Qdrant.
        top_doc = raw_results[0]
        top_score = top_doc.get("cross_encoder_score", top_doc.get("score", 0.0))
        RAG_TOP_SCORE.labels(source="rest").observe(top_score)

    documents = [Document(score=r["score"], metadata=r["metadata"]) for r in raw_results]
    return SearchResponse(
        results=documents,
        total=n_results,
        query_time_ms=round(query_time_ms, 2),
    )
