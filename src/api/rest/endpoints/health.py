from fastapi import APIRouter


router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check():
    """Эндпоинт для Kubernetes / Docker Healthcheck."""
    return {"status": "ok", "message": "ML API is running"}
