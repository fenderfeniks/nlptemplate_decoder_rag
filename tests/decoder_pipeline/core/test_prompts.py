# tests/decoder_pipeline/core/test_prompts.py
import pytest

from src.decoder_pipeline.core.prompts.manager import PromptManager


class TestPromptManager:
    @pytest.fixture
    def manager(self):
        templates = {
            "qa": "Вопрос: {{ question }}\nКонтекст: {{ context }}\nОтвет:",
            "simple": "Сделай саммари: {{ text }}",
        }
        return PromptManager(templates=templates)

    def test_renders_template_successfully(self, manager):
        result = manager.render("qa", question="Как дела?", context="Все хорошо.")
        assert "Вопрос: Как дела?" in result
        assert "Контекст: Все хорошо." in result

    def test_missing_template_raises(self, manager):
        """Вызов неизвестного шаблона должен падать с ValueError."""
        with pytest.raises(ValueError, match="не найден в реестре"):
            manager.render("unknown_template", question="test")

    def test_missing_variables_raises(self, manager):
        """Если переданы не все переменные шаблона, менеджер должен упасть."""
        with pytest.raises(ValueError, match="Пропущены обязательные переменные"):
            # Забыли передать 'context'
            manager.render("qa", question="Только вопрос")
