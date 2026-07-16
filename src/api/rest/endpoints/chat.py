from fastapi import APIRouter, Depends, HTTPException
import logging

from src.api.schemas import ChatRequest, ChatResponse
from src.api.rest.dependencies import get_generator, get_retriever, get_prompt_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Generation"])

@router.post("/generate", response_model=ChatResponse)
async def generate_text(
    request: ChatRequest,
    generator = Depends(get_generator),
    retriever = Depends(get_retriever),
    prompt_manager = Depends(get_prompt_manager)  # Наш заменитель LangChain Prompts
):
    try:
        context = None
        
        # 1. Формируем историю диалога (если она есть)
        history_text = ""
        if request.history:
            history_text = "История предыдущего диалога:\n"
            for msg in request.history:
                history_text += f"{msg.role.capitalize()}: {msg.content}\n"
            history_text += "\n"

        # 2. Идем в базу RAG (если просят)
        if request.use_rag:
            context = retriever.retrieve_context(request.query)
            # Используем PromptManager для красивой сборки промпта
            final_prompt = prompt_manager.build_rag_prompt(
                query=request.query, 
                context=context,
                history=history_text # В PromptManager нужно будет добавить этот аргумент
            )
        else:
            final_prompt = prompt_manager.build_simple_prompt(request.query)

        # 3. Инференс (Сам генератор внутри уже использует ResponseCleaner!)
        if request.max_tokens:
            generator.generation_kwargs["max_new_tokens"] = request.max_tokens

        responses = generator.generate(final_prompt)
        
        return ChatResponse(
            answer=responses[0],
            context_used=context
        )

    except Exception as e:
        logger.error(f"Ошибка инференса: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка генерации ответа.")