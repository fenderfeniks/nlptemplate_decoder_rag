from fastapi import HTTPException, Request

from src.application.orchestrator import RAGOrchestrator


def get_orchestrator(request: Request) -> RAGOrchestrator:
    orchestrator = request.app.state.orchestrator
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Оркестратор не готов.")
    return orchestrator
