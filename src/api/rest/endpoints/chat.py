import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

# --- 1. ДОБАВЛЯЕМ ИМПОРТ НАШИХ МЕТРИК ---
from src.api.metrics import LLM_GENERATIONS_TOTAL, LLM_INFERENCE_TIME
from src.api.rest.dependencies import get_generator, get_prompt_manager, get_retriever
from src.api.schemas import ChatRequest, ChatResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Generation"])


@router.post("/generate", response_model=ChatResponse)
async def generate_text(
    request: ChatRequest,
    generator=Depends(get_generator),
    retriever=Depends(get_retriever),
    prompt_manager=Depends(get_prompt_manager),  # Наш заменитель LangChain Prompts
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

        # ИСПРАВЛЕНИЕ: Склеиваем историю с запросом, чтобы не прокидывать kwarg history 
        # и избежать TypeError при вызове build_rag_prompt
        full_query = history_text + request.query

        # 2. Идем в базу RAG (если просят)
        if request.use_rag:
            # ИСПРАВЛЕНИЕ: Оборачиваем синхронный вызов к RAG в to_thread (защита Event Loop)
            context = await asyncio.to_thread(retriever.retrieve_context, request.query)
            
            # Используем PromptManager для красивой сборки промпта
            final_prompt = prompt_manager.build_rag_prompt(
                query=full_query,
                context=context,
            )
        else:
            final_prompt = prompt_manager.build_simple_prompt(full_query)

        # 3. Инференс
        # ИСПРАВЛЕНИЕ: Формируем kwargs локально, чтобы избежать Race Condition (не мутируем shared state)
        local_gen_kwargs = {}
        if request.max_tokens:
            local_gen_kwargs["max_new_tokens"] = request.max_tokens

        # --- 2. ДОБАВЛЯЕМ СБОР МЕТРИК ---

        # Увеличиваем счетчик запросов именно из REST API
        LLM_GENERATIONS_TOTAL.labels(source="rest").inc()

        # Засекаем чистое время работы нейросети
        with LLM_INFERENCE_TIME.labels(source="rest").time():
            # ИСПРАВЛЕНИЕ: Выносим тяжелую генерацию в отдельный поток (защита Event Loop)
            responses = await asyncio.to_thread(
                generator.generate, 
                final_prompt, 
                **local_gen_kwargs
            )

        return ChatResponse(answer=responses[0], context_used=context)

    except Exception as e:
        logger.error(f"Ошибка инференса: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка генерации ответа.") from e