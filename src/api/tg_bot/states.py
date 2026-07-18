"""
Машина состояний (FSM) для Telegram-бота.
"""

from aiogram.fsm.state import State, StatesGroup


class ChatProcess(StatesGroup):
    """
    Базовые состояния пользователя в боте.
    """

    chatting = State()  # Обычный режим общения
    waiting_for_document = State()  # Резерв на будущее (если захотим заливать PDF через бота)
