# src/pipelines/rag/api/rest/dependencies.py
from fastapi import HTTPException, Request

from src.pipelines.rag.inference.retriever import HybridRetriever


def get_retriever(request: Request) -> HybridRetriever:
    """FastAPI Depends — возвращает инициализированный HybridRetriever.

    Реранкер встроен в HybridRetriever: если он был передан при создании
    (cfg.retrieval.reranker != null), он применяется автоматически
    внутри retriever.search(). Отдельный Depends для реранкера не нужен.
    """
    retriever = request.app.state.ml_models.get("retriever")
    if not retriever:
        raise HTTPException(status_code=503, detail="Ретривер не инициализирован.")
    return retriever
