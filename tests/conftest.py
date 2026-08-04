# conftest.py
import pytest
from datasets import Dataset


@pytest.fixture
def sample_text_dataset() -> Dataset:
    """Базовый датасет с дубликатами для тестирования дедупликации и очистки."""
    return Dataset.from_dict(
        {
            "text": [
                "Пример текста 1",
                "Пример текста 1",  # Точный дубликат
                "Пример текста 2",
                "пример текста 1 ",  # Нечеткий дубликат (регистр, пробел)
            ],
            "prompt": ["p1", "p2", "p3", "p4"],
            "metadata": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def sample_tokenized_dataset() -> Dataset:
    """Датасет с токенами для тестирования фильтрации по длине."""
    return Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3],  # length 3
                [1, 2, 3, 4, 5, 6, 7],  # length 7
                [1, 2],  # length 2
            ],
            "text": ["short", "too long", "very short"],
        }
    )
