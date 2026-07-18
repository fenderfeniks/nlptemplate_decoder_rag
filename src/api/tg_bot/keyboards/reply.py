"""
Reply-клавиатуры (нижние кнопки).
"""

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Главная клавиатура бота."""
    keyboard = [
        [KeyboardButton(text="🧹 Очистить контекст")],
        [KeyboardButton(text="⚙️ RAG: Вкл"), KeyboardButton(text="⚙️ RAG: Выкл")],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,  # Кнопки будут компактными
        input_field_placeholder="Введите ваш вопрос...",
    )
