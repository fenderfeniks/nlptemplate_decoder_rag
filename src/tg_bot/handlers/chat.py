import asyncio
import logging
import time
from contextlib import suppress

from aiogram import F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from omegaconf import DictConfig

from src.application.orchestrator import RAGOrchestrator
from src.decoder_pipeline.core.prompts.manager import PromptManager
from src.tg_bot.keyboards.reply import get_main_keyboard
from src.tg_bot.states import ChatProcess


logger = logging.getLogger(__name__)
router = Router()


@router.message(CommandStart())
async def cmd_start(
    message: types.Message,
    state: FSMContext,
    cfg: DictConfig,
    prompt_manager: PromptManager,
) -> None:
    await state.set_state(ChatProcess.chatting)
    user_name = message.from_user.full_name if message.from_user else "Пользователь"
    welcome_text = prompt_manager.render("telegram_welcome", username=user_name)
    await message.answer(text=welcome_text, reply_markup=get_main_keyboard())


@router.message(ChatProcess.chatting, F.text)
async def process_chat_message(
    message: types.Message,
    cfg: DictConfig,
    orchestrator: RAGOrchestrator,
) -> None:
    query = message.text or ""

    tg_cfg = cfg.tg_bot
    processing_text = tg_cfg.messages.get("processing", "✨ Генерирую ответ...")
    error_text = tg_cfg.messages.get("error", "Произошла ошибка. Попробуйте ещё раз.")
    edit_interval: float = tg_cfg.get("stream_edit_interval", 1.5)
    top_k: int = tg_cfg.get("top_k", 5)

    processing_msg = await message.answer(processing_text)

    async def _generate() -> None:
        try:
            start_time = time.perf_counter()
            answer = ""
            last_edit_time = time.time()

            # Асинхронно читаем стрим из сетевого оркестратора
            async for chunk in orchestrator.ask_stream(query=query, top_k=top_k):
                answer += chunk
                if time.time() - last_edit_time > edit_interval:
                    with suppress(TelegramBadRequest):
                        await processing_msg.edit_text(f"{answer} ⏳")
                    last_edit_time = time.time()

            elapsed = time.perf_counter() - start_time
            final_text = f"{answer}\n\n_⏱ {elapsed:.1f} сек._"

            with suppress(TelegramBadRequest):
                await processing_msg.edit_text(final_text, parse_mode="Markdown")

        except Exception:
            logger.error("Ошибка при генерации ответа", exc_info=True)
            with suppress(TelegramBadRequest):
                await processing_msg.edit_text(error_text)

    asyncio.create_task(_generate())
