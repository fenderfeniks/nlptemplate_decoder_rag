"""CLI-утилиты для Hydra-скриптов.

enforce_pipeline — гарантирует что sys.argv содержит правильный pipeline_name
перед передачей управления Hydra. Без этого Hydra берёт дефолт из main.yaml,
что при запуске RAG-скриптов из общего окружения даёт неверную конфигурацию.
"""

import logging
import sys


logger = logging.getLogger(__name__)


def enforce_pipeline(expected: str, *extra_overrides: str) -> None:
    """Гарантировать ``pipeline_name=<expected>`` в ``sys.argv``.

    Вызывать в ``if __name__ == '__main__'`` до точки входа Hydra.

    Args:
        expected: Ожидаемое значение pipeline_name, например ``'rag_pipeline'``.
        *extra_overrides: Дополнительные Hydra-оверрайды, которые нужно добавить
            если они ещё не присутствуют в ``sys.argv``. Проверка по префиксу
            до первого ``=``. Например: ``'rag_pipeline/data=indexing'``.

    Example::

        if __name__ == '__main__':
            enforce_pipeline("rag_pipeline", "rag_pipeline/data=indexing")
            index_database()
    """
    arg_prefix = "pipeline_name="
    idx = next((i for i, a in enumerate(sys.argv) if a.startswith(arg_prefix)), None)

    if idx is not None:
        current = sys.argv[idx].split("=", 1)[1]
        if current != expected:
            logger.warning(
                "ВНИМАНИЕ! Передано pipeline_name=%s. Принудительно меняем на '%s'.",
                current,
                expected,
            )
            sys.argv[idx] = f"{arg_prefix}{expected}"
    else:
        sys.argv.append(f"{arg_prefix}{expected}")

    for override in extra_overrides:
        prefix = override.split("=", 1)[0]
        if not any(a.startswith(f"{prefix}=") for a in sys.argv):
            logger.info("Добавляем Hydra-оверрайд: %s", override)
            sys.argv.append(override)
