from fastapi import HTTPException, Request

from src.pipelines.rag.retrieval.retriever import BaseRetriever


def get_retriever(request: Request) -> BaseRetriever:
    retriever = request.app.state.ml_models.get("retriever")
    if not retriever:
        raise HTTPException(status_code=503, detail="Ретривер не инициализирован.")
    return retriever
