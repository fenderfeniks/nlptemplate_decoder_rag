# tests/pipelines/decoder/core/prompts/test_manager.py
import pytest
from omegaconf import OmegaConf

from src.pipelines.decoder.core.prompts.manager import PromptManager


class TestPromptManager:
    def test_init_with_dict(self):
        """Инициализация через обычный словарь."""
        templates = {"test_prompt": "Hello {{ name }}!"}
        manager = PromptManager(templates=templates)
        assert manager.templates == templates

    def test_init_with_dictconfig(self):
        """Инициализация через DictConfig от Hydra."""
        cfg = OmegaConf.create({"test_prompt": "Hello {{ name }}!"})
        manager = PromptManager(templates=cfg)
        assert manager.templates == {"test_prompt": "Hello {{ name }}!"}

    def test_init_with_none(self):
        """Если передать None, создается пустой реестр."""
        manager = PromptManager(templates=None)
        assert manager.templates == {}

    def test_successful_render(self):
        """Успешный рендеринг промпта."""
        manager = PromptManager(templates={"greet": "Hello {{ name }}! You are {{ age }}."})
        result = manager.render("greet", name="Alice", age=25)
        assert result == "Hello Alice! You are 25."

    def test_render_missing_template(self):
        """Ошибка, если шаблон не найден."""
        manager = PromptManager(templates={})
        with pytest.raises(ValueError, match="не найден в реестре"):
            manager.render("unknown_template")

    def test_render_missing_variable(self):
        """Ошибка, если переданы не все переменные для шаблона."""
        manager = PromptManager(templates={"greet": "Hello {{ name }}!"})
        # Забыли передать name
        with pytest.raises(ValueError, match="Пропущены обязательные переменные"):
            manager.render("greet", something_else=123)
