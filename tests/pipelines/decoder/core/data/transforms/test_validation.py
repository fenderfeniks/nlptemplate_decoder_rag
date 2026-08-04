import pytest
from datasets import Dataset

from src.pipelines.decoder.core.data.transforms.validation import DecoderValidationTransform


class TestDecoderValidationTransform:
    def test_invalid_mode(self):
        """Проверка исключения при неизвестном режиме."""
        with pytest.raises(ValueError, match="Неизвестный режим"):
            DecoderValidationTransform(mode="unknown_mode")

    def test_cpt_validation_methods_direct(self):
        """Прямое тестирование логики CPT без HF datasets (для точного трекинга coverage)."""
        transform = DecoderValidationTransform(mode="cpt")
        
        # Проверяем служебные методы
        assert transform._get_required_columns() == ["text"]
        assert transform._get_filter_column() == "text"
        
        # Проверяем саму логику валидации батча
        batch = {
            "text": [
                "Нормальный длинный текст", 
                "Да",   # Слишком короткий
                None    # Нельзя None
            ]
        }
        res = transform._validate_batch(batch)
        
        # Pydantic должен заменить битые записи на пустые строки
        assert res["text"] == ["Нормальный длинный текст", "", ""]

    def test_sft_validation_methods_direct(self):
        """Прямое тестирование логики SFT без HF datasets (для точного трекинга coverage)."""
        transform = DecoderValidationTransform(mode="sft")
        
        # Проверяем служебные методы
        assert transform._get_required_columns() == ["prompt", "target"]
        assert transform._get_filter_column() == "prompt"
        
        # Проверяем саму логику валидации батча
        batch = {
            "prompt": ["Промпт 1", "П", "Промпт 3"],
            "target": ["Ответ 1", "Ответ 2", "   "] # Пустой таргет
        }
        res = transform._validate_batch(batch)
        
        # Записи с индексом 1 (короткий промпт) и 2 (пустой таргет) должны обнулиться
        assert res["prompt"] == ["Промпт 1", "", ""]
        assert res["target"] == ["Ответ 1", "", ""]

    def test_full_pipeline_cpt(self):
        """Интеграционный тест: проверяем, что __call__ корректно фильтрует HF Dataset."""
        ds = Dataset.from_dict({"text": ["Нормальный текст", "Да"]})
        transform = DecoderValidationTransform(mode="cpt", num_proc=1)
        
        result = transform(ds)
        # "Да" должно быть отфильтровано
        assert len(result) == 1
        assert result["text"] == ["Нормальный текст"]