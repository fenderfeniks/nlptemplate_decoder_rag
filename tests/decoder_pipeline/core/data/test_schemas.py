# tests/decoder_pipeline/core/data/test_schemas.py
import pytest
from pydantic import ValidationError

from src.decoder_pipeline.core.data.schemas import RawDatasetRecord


class TestRawDatasetRecord:
    def test_valid_cpt_record(self):
        record = RawDatasetRecord(text="Это валидный длинный текст для претрейна.")
        assert record.text == "Это валидный длинный текст для претрейна."
        assert record.prompt is None
        assert record.target is None

    def test_valid_sft_record(self):
        record = RawDatasetRecord(prompt="Привет, как дела?", target="Все отлично!")
        assert record.prompt == "Привет, как дела?"
        assert record.target == "Все отлично!"
        assert record.text is None

    def test_missing_all_required_fields_raises(self):
        with pytest.raises(ValidationError, match="Должен быть заполнен либо 'text'"):
            RawDatasetRecord()

    def test_too_short_text_raises(self):
        with pytest.raises(ValidationError, match="Текст слишком короткий"):
            RawDatasetRecord(text="а")

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError, match="Поле ввода не может быть пустым"):
            RawDatasetRecord(text="   ")

    def test_empty_target_raises(self):
        with pytest.raises(ValidationError, match="Target передан, но является пустой строкой"):
            RawDatasetRecord(prompt="Нормальный промпт", target="   ")