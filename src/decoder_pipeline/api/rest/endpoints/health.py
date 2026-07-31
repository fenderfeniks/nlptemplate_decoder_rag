# src/api/rest/endpoints/health.py
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse


router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """Эндпоинт для Kubernetes / Docker Healthcheck.

    Возвращает 200 только если все критичные компоненты инициализированы.
    Kubernetes readiness probe должна использовать этот эндпоинт.
    """
    issues: list[str] = []

    # Проверяем LLM-клиент
    ml_models = getattr(request.app.state, "ml_models", {})
    if not ml_models.get("generator"):
        issues.append("LLM generator не инициализирован")

    # Проверяем PromptManager
    if not getattr(request.app.state, "prompt_manager", None):
        issues.append("PromptManager не инициализирован")

    if issues:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "issues": issues},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "message": "ML API is running"},
    )
