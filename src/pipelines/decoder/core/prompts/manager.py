# src/pipelines/decoder/core/prompts/manager.py
import logging
from typing import Any

from jinja2 import Environment, meta
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


class PromptManager:
    """Индустриальный менеджер промптов с поддержкой композитных структур.

    Обеспечивает безопасный рендеринг Jinja2-шаблонов. Поддерживает как
    монолитные строковые шаблоны, так и блочные (словарь секций).
    """

    def __init__(
        self,
        templates: dict[str, Any] | DictConfig | None = None,
        default_order: list[str] | None = None,
    ) -> None:
        """
        Args:
            templates: Словарь с шаблонами или DictConfig от Hydra.
            default_order: Порядок сборки блоков по умолчанию для композитных
                промптов. Если None, используется стандартный пайплайн.
        """
        if isinstance(templates, DictConfig):
            self.templates = OmegaConf.to_container(templates, resolve=True)
        else:
            self.templates = templates or {}

        self.env = Environment(autoescape=False)
        
        # Стандартный индустриальный порядок сборки, если в конфиге не указано иное
        self.default_order = default_order or [
            "role",
            "context",
            "task",
            "constraints",
            "examples",
            "thought_process",
            "input_format",
            "output_format",
        ]
        logger.info("PromptManager готов. Загружено шаблонов: %d", len(self.templates))

    def _render_string(self, template_str: str, template_name: str, **kwargs: Any) -> str:
        ast = self.env.parse(template_str)
        required_vars = meta.find_undeclared_variables(ast)

        # find_undeclared_variables возвращает ВСЕ переменные включая те,
        # что внутри {% if var %}...{% endif %} — они фактически опциональны.
        # Передаём None для отсутствующих — Jinja корректно вычислит {% if None %} как False.
        kwargs_with_defaults = {
            var: kwargs.get(var, None) for var in required_vars
        }
        kwargs_with_defaults.update(kwargs)  # явно переданные имеют приоритет

        template = self.env.from_string(template_str)
        return template.render(**kwargs_with_defaults)

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Рендерит промпт по имени, подставляя переменные.

        Если шаблон является словарем, собирает его из блоков с учетом
        указанного порядка (или порядка по умолчанию), пропуская пустые.

        Args:
            template_name: Имя шаблона из загруженной конфигурации.
            **kwargs: Значения для переменных шаблона.

        Returns:
            Собранная строка промпта.
        """
        if template_name not in self.templates:
            raise ValueError(
                f"Шаблон '{template_name}' не найден в реестре. "
                f"Доступные: {list(self.templates.keys())}"
            )

        template_data = self.templates[template_name]

        # 1. Обратная совместимость: если это просто строка
        if isinstance(template_data, str):
            rendered = self._render_string(template_data, template_name, **kwargs)
            logger.debug("Отрендерен строковой промпт '%s'", template_name)
            return rendered

        # 2. Композитный промпт: сборка из блоков
        if isinstance(template_data, dict):
            parts = []
            # Позволяем конфигу переопределить порядок через ключ __order__
            block_order = template_data.get("__order__", self.default_order)

            for block_name in block_order:
                block_content = template_data.get(block_name)
                
                if not block_content or not isinstance(block_content, str):
                    continue

                rendered_block = self._render_string(
                    block_content, 
                    f"{template_name}.{block_name}", 
                    **kwargs
                )
                parts.append(rendered_block.strip())

            final_prompt = "\n\n".join(parts)
            logger.debug("Отрендерен композитный промпт '%s'", template_name)
            return final_prompt

        raise TypeError(
            f"Шаблон '{template_name}' имеет неверный тип ({type(template_data)}). "
            "Ожидается str или dict."
        )