# src/rag_pipeline/api/rest/endpoints/health.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["System"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Liveness probe — сервис жив и принимает соединения.

    Kubernetes перезапускает Pod если этот эндпоинт недоступен.
    Не проверяет состояние моделей — только что процесс работает.
    """
    return {"status": "ok", "service": "rag_api"}


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """Readiness probe — сервис готов принимать трафик.

    Kubernetes не направляет запросы на Pod пока этот эндпоинт возвращает не-200.
    Проверяет, что RAG-стек полностью загружен и индекс не пуст.
    """
    retriever = request.app.state.ml_models.get("retriever")

    if retriever is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "retriever not initialized"},
        )

    ntotal = retriever.vector_db.index.ntotal
    if ntotal == 0:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "vector index is empty"},
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready", "indexed_documents": ntotal},
    )


# Оставляем /health как алиас liveness для обратной совместимости с Docker healthcheck
@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "rag_api"}
