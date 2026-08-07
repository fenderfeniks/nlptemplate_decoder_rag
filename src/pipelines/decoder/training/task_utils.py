"""Вспомогательные утилиты для декодер-пайплайна.

resolve_task_mode — определение режима обучения (sft / cpt) из конфига данных.
Вынесено из train.py чтобы логика не дублировалась если понадобится в eval.
"""

from omegaconf import DictConfig


def resolve_task_mode(data_cfg: DictConfig | dict) -> str:
    """Определить task_mode (``'sft'`` или ``'cpt'``) из конфига данных.

    Приоритет:
        1. Явный ``data_cfg.task in ['sft', 'cpt']`` — берём как есть.
        2. Есть ``prompt_column`` — SFT (инструкционный файнтюн).
        3. Нет ``prompt_column`` — CPT (continued pretraining).

    Args:
        data_cfg: Секция ``cfg.decoder_pipeline.data`` (DictConfig или dict).

    Returns:
        ``'sft'`` или ``'cpt'``.
    """
    _get = (
        (lambda k: data_cfg.get(k))
        if isinstance(data_cfg, dict)
        else (lambda k: getattr(data_cfg, k, None))
    )

    task_val = _get("task")
    if task_val in ("sft", "cpt"):
        return task_val

    return "sft" if bool(_get("prompt_column")) else "cpt"
