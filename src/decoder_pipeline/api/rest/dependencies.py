from fastapi import HTTPException, Request

from src.decoder_pipeline.core.prompts.manager import PromptManager
from src.decoder_pipeline.sdk.inference import LLMGenerationClient


def get_prompt_manager(request: Request) -> PromptManager:
    return request.app.state.prompt_manager


def get_generator(request: Request) -> LLMGenerationClient:
    generator = request.app.state.ml_models.get("generator")
    if not generator:
        raise HTTPException(status_code=503, detail="LLM клиент не инициализирован.")
    return generator
