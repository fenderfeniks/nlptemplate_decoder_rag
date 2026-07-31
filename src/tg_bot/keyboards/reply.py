# src/tg_bot/keyboards/reply.py
"""Reply-клавиатуры (нижние кнопки)."""

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Создаёт главную клавиатуру бота.

    Returns:
        Инициализированный объект ReplyKeyboardMarkup.
    """
    keyboard = [
        [KeyboardButton(text="/start")],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        input_field_placeholder="Задайте вопрос — найду ответ в базе знаний...",
    )
