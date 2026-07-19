"""
Главный обработчик сообщений (Бизнес-логика бота).
Универсален: умеет ходить по HTTP (для bot_local) и напрямую в память (для bot_webhook).
"""

import asyncio
import logging
import os

import aiohttp
from aiogram import F, Router, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext

# --- 1. ДОБАВЛЯЕМ ИМПОРТ МЕТРИК ---
from src.api.metrics import LLM_GENERATIONS_TOTAL, LLM_INFERENCE_TIME
from src.api.tg_bot.keyboards.reply import get_main_keyboard
from src.api.tg_bot.states import ChatProcess


logger = logging.getLogger(__name__)

# Создаем роутер (аналог APIRouter в FastAPI)
router = Router()


@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext, cfg):
    """Обработка команды /start."""
    await state.set_state(ChatProcess.chatting)
    # Сохраняем настройку RAG по умолчанию в состояние юзера
    await state.update_data(use_rag=cfg.api.telegram.default_use_rag)

    await message.answer(text=cfg.api.telegram.messages.welcome, reply_markup=get_main_keyboard())


@router.message(F.text == "🧹 Очистить контекст")
async def clear_context(message: types.Message, state: FSMContext):
    data = await state.get_data()
    await state.clear()
    await state.set_state(ChatProcess.chatting)
    # Возвращаем настройку RAG и создаем пустую историю
    await state.update_data(use_rag=data.get("use_rag", True), history=[])

    await message.answer("Контекст диалога успешно очищен! 🧹", reply_markup=get_main_keyboard())


@router.message(ChatProcess.chatting, F.text)
async def process_chat_message(
    message: types.Message,
    state: FSMContext,
    cfg,
    generator=None,
    retriever=None,
    prompt_manager=None,
    api_url=None,
):
    if message.text == "⚙️ RAG: Вкл":
        await state.update_data(use_rag=True)
        return await message.answer("Поиск по базе знаний ВКЛЮЧЕН 🔍")
    elif message.text == "⚙️ RAG: Выкл":
        await state.update_data(use_rag=False)
        return await message.answer("Поиск по базе знаний ВЫКЛЮЧЕН ❌")

    processing_msg = await message.answer(cfg.api.telegram.messages.thinking)
    user_data = await state.get_data()
    use_rag = user_data.get("use_rag", True)

    # ИСПРАВЛЕНИЕ: Достаем историю диалога
    history = user_data.get("history", [])

    try:
        # --- СЦЕНАРИЙ А: ПРОДАКШЕН (Webhooks) ---
        if generator and retriever and prompt_manager:
            # Склеиваем историю сообщений руками, как мы это делали в FastAPI
            history_text = ""
            if history:
                history_text = "История предыдущего диалога:\n"
                for msg in history:
                    history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
                history_text += "\n"

            full_query = history_text + message.text

            context = None
            if use_rag:
                context = await asyncio.to_thread(retriever.retrieve_context, message.text)
                prompt = prompt_manager.build_rag_prompt(full_query, context)
            else:
                prompt = prompt_manager.build_simple_prompt(full_query)

            LLM_GENERATIONS_TOTAL.labels(source="tg").inc()
            with LLM_INFERENCE_TIME.labels(source="tg").time():
                responses = await asyncio.to_thread(generator.generate, prompt)
            answer = responses[0]

        # --- СЦЕНАРИЙ Б: ЛОКАЛЬНАЯ РАЗРАБОТКА (Polling) ---
        elif api_url:
            payload = {
                "query": message.text,
                "history": history,  # Прокидываем историю в API
                "use_rag": use_rag,
                "max_tokens": cfg.api.telegram.max_tokens,
            }
            # ИСПРАВЛЕНИЕ: Прокидываем API-ключ
            headers = {"X-API-Key": os.getenv("API_KEY", "")}

            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        answer = data.get("answer", cfg.api.telegram.messages.error)
                    else:
                        answer = f"Ошибка API: {resp.status}"
        else:
            raise ValueError("Не передан ни generator, ни api_url!")

        await processing_msg.edit_text(answer)

        # ИСПРАВЛЕНИЕ: Обновляем и сохраняем историю в Redis/Memory (последние 10 сообщений)
        history.append({"role": "user", "content": message.text})
        history.append({"role": "assistant", "content": answer})
        await state.update_data(history=history[-10:])

    except Exception as e:
        logger.error(f"Ошибка в хэндлере ТГ: {str(e)}")
        await processing_msg.edit_text(cfg.api.telegram.messages.error)
