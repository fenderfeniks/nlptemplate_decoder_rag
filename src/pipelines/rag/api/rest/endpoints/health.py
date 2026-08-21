# src/pipelines/rag/api/rest/endpoints/health.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from src.pipelines.rag.api.metrics import RAG_INDEX_TOTAL_DOCS

router = APIRouter(tags=["System"])


@router.get("/health/live")
async def liveness(request: Request) -> dict[str, str]:
    """Liveness probe — сервис жив и принимает соединения.

    Kubernetes перезапускает Pod если этот эндпоинт недоступен.
    Не проверяет состояние моделей — только что процесс работает.
    """
    return {"status": "ok", "service": request.app.state.service_name}


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """Readiness probe — сервис готов принимать трафик.

    Kubernetes не направляет запросы на Pod пока этот эндпоинт возвращает не-200.
    Проверяет что RAG-стек полностью загружен и индекс не пуст.
    """
    retriever = request.app.state.ml_models.get("retriever")

    if retriever is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "retriever not initialized"},
        )

    # Используем .ntotal из BaseVectorStore протокола — не лезем в .index.ntotal
    # напрямую. Это позволяет readiness probe работать с любым бэкендом (Qdrant и т.д.)
    ntotal = retriever.vector_db.ntotal
    RAG_INDEX_TOTAL_DOCS.set(ntotal)
    if ntotal == 0:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "vector index is empty"},
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready", "indexed_documents": ntotal},
    )


@router.get("/health")
async def health_check(request: Request) -> dict[str, str]:
    """Алиас liveness для обратной совместимости с Docker healthcheck."""
    return {"status": "ok", "service": request.app.state.service_name}
