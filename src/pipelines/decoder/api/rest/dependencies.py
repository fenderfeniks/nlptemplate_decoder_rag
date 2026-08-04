# src/api/rest/dependencies.py
from fastapi import HTTPException, Request

from src.pipelines.decoder.core.prompts.manager import PromptManager
from src.pipelines.decoder.inference.inference import LLMGenerationClient


def get_prompt_manager(request: Request) -> PromptManager:
    manager: PromptManager | None = getattr(request.app.state, "prompt_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="PromptManager не инициализирован.")
    return manager


def get_generator(request: Request) -> LLMGenerationClient:
    generator: LLMGenerationClient | None = request.app.state.ml_models.get("generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="LLM клиент не инициализирован.")
    return generator
