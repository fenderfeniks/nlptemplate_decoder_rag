# tests/pipelines/decoder/core/data/test_schemas.py
import pytest
from pydantic import ValidationError

from src.pipelines.decoder.core.data.schemas import RawDatasetRecord


class TestRawDatasetRecord:
    def test_valid_cpt_record(self):
        """Успешное создание записи для Continual Pre-Training."""
        record = RawDatasetRecord(text="Пример длинного текста")
        assert record.text == "Пример длинного текста"

    def test_valid_sft_record(self):
        """Успешное создание записи для Supervised Fine-Tuning."""
        record = RawDatasetRecord(prompt="Вопрос?", target="Ответ!")
        assert record.prompt == "Вопрос?"
        assert record.target == "Ответ!"

    def test_missing_required_fields(self):
        """Ошибка, если не передан ни text, ни prompt."""
        with pytest.raises(ValidationError, match="Должен быть заполнен либо 'text'"):
            RawDatasetRecord()

    def test_text_too_short(self):
        """Ошибка при слишком коротком тексте (фильтрация мусора)."""
        with pytest.raises(ValidationError, match="Текст слишком короткий"):
            RawDatasetRecord(text="Да")

    def test_empty_target(self):
        """Ошибка при пустом таргете для SFT."""
        with pytest.raises(ValidationError, match="Target передан, но является пустой строкой"):
            RawDatasetRecord(prompt="Вопрос?", target="   ")