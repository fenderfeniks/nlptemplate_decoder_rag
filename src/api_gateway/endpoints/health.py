# src/api_gateway/endpoints/health.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["System"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Liveness probe — процесс жив и принимает соединения.

    Kubernetes перезапускает Pod если этот эндпоинт недоступен.
    Не проверяет состояние моделей или downstream-сервисов.
    """
    return {"status": "ok", "service": "api-gateway"}


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """Readiness probe — Gateway готов принимать трафик.

    Kubernetes не направляет запросы на Pod пока этот эндпоинт возвращает не-200.
    Проверяет:
        1. Оркестратор инициализирован (lifespan завершён).
        2. LLM сервер отвечает на /health (llama.cpp / vLLM).

    Не проверяет RAG API — он имеет собственный circuit breaker и
    graceful degradation: недоступность RAG не делает Gateway не-ready.
    """
    import httpx

    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "orchestrator not initialized"},
        )

    # Проверяем LLM сервер — без него генерация невозможна
    llm_api_base = orchestrator.llm_client.api_base.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{llm_api_base}/health")
            resp.raise_for_status()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": f"llm server unavailable: {type(e).__name__}",
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "llm_model": orchestrator.llm_client.model_name,
        },
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Алиас liveness для совместимости с Docker healthcheck и e2e eval."""
    return {"status": "ok", "service": "api-gateway"}
