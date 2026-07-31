# src/api_gateway/dependencies.py
from fastapi import Depends, HTTPException, Request

from src.application.orchestrator import RAGOrchestrator


def get_orchestrator(request: Request) -> RAGOrchestrator:
    """FastAPI dependency: возвращает инициализированный RAGOrchestrator из app.state.

    Raises:
        HTTPException 503: если оркестратор ещё не готов (lifespan не завершён).
    """
    orchestrator: RAGOrchestrator | None = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Оркестратор не готов.")
    return orchestrator
