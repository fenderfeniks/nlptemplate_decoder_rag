"""
Главный обработчик сообщений (Бизнес-логика бота).
Универсален: умеет ходить по HTTP (для bot_local) и напрямую в память (для bot_webhook).
"""
import aiohttp
import logging
from aiogram import Router, F, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext

from src.api.tg_bot.keyboards.reply import get_main_keyboard
from src.api.tg_bot.states import ChatProcess

# --- 1. ДОБАВЛЯЕМ ИМПОРТ МЕТРИК ---
from src.api.metrics import LLM_GENERATIONS_TOTAL, LLM_INFERENCE_TIME

logger = logging.getLogger(__name__)

# Создаем роутер (аналог APIRouter в FastAPI)
router = Router()

@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext, cfg):
    """Обработка команды /start."""
    await state.set_state(ChatProcess.chatting)
    # Сохраняем настройку RAG по умолчанию в состояние юзера
    await state.update_data(use_rag=cfg.api.telegram.default_use_rag)
    
    await message.answer(
        text=cfg.api.telegram.messages.welcome,
        reply_markup=get_main_keyboard()
    )


@router.message(F.text == "🧹 Очистить контекст")
async def clear_context(message: types.Message, state: FSMContext):
    """Кнопка очистки истории."""
    # Очищаем историю, но оставляем настройку RAG
    data = await state.get_data()
    await state.clear()
    await state.set_state(ChatProcess.chatting)
    await state.update_data(use_rag=data.get("use_rag", True))
    
    await message.answer("Контекст диалога успешно очищен! 🧹", reply_markup=get_main_keyboard())


@router.message(ChatProcess.chatting, F.text)
async def process_chat_message(
    message: types.Message, 
    state: FSMContext, 
    cfg, 
    # Эти параметры могут быть None, зависимо от того, кто вызвал хэндлер
    generator=None, 
    retriever=None,
    prompt_manager=None,
    api_url=None
):
    """Универсальный генератор ответов."""
    
    # 1. Ловим клики по кнопкам переключения RAG
    if message.text == "⚙️ RAG: Вкл":
        await state.update_data(use_rag=True)
        return await message.answer("Поиск по базе знаний ВКЛЮЧЕН 🔍")
    elif message.text == "⚙️ RAG: Выкл":
        await state.update_data(use_rag=False)
        return await message.answer("Поиск по базе знаний ВЫКЛЮЧЕН ❌")

    processing_msg = await message.answer(cfg.api.telegram.messages.thinking)
    user_data = await state.get_data()
    use_rag = user_data.get("use_rag", True)

    try:
        # =================================================================
        # СЦЕНАРИЙ А: ПРОДАКШЕН (Webhooks). Идем напрямую в видеокарту.
        # =================================================================
        if generator and retriever and prompt_manager:
            context = None
            if use_rag:
                context = retriever.retrieve_context(message.text)
                prompt = prompt_manager.build_rag_prompt(message.text, context)
            else:
                prompt = prompt_manager.build_simple_prompt(message.text)

            # --- 2. СБОР МЕТРИК ДЛЯ ТЕЛЕГРАМ-БОТА ---
            LLM_GENERATIONS_TOTAL.labels(source="tg").inc()
            
            with LLM_INFERENCE_TIME.labels(source="tg").time():
                responses = generator.generate(prompt)
                
            answer = responses[0]

        # =================================================================
        # СЦЕНАРИЙ Б: ЛОКАЛЬНАЯ РАЗРАБОТКА (Polling). Идем по HTTP в FastAPI.
        # =================================================================
        elif api_url:
            payload = {
                "query": message.text,
                "use_rag": use_rag,
                "max_tokens": cfg.api.telegram.max_tokens
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        answer = data.get("answer", cfg.api.telegram.messages.error)
                    else:
                        answer = f"Ошибка API: {resp.status}"
        else:
            raise ValueError("Не передан ни generator, ни api_url!")

        # Отправляем финальный ответ
        await processing_msg.edit_text(answer)

    except Exception as e:
        logger.error(f"Ошибка в хэндлере ТГ: {str(e)}")
        await processing_msg.edit_text(cfg.api.telegram.messages.error)