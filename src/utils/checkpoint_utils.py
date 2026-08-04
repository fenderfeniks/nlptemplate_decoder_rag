# src/utils/checkpoint_utils.py
import logging
from pathlib import Path

import torch
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


def load_checkpoint(
    model: PreTrainedModel,
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> PreTrainedModel:
    """Загружает веса в базовую модель из файла или директории.

    Поддерживает три формата:
    - Директория с ``adapter_config.json``  → загружается как LoRA (PeftModel).
    - Директория с ``pytorch_model.bin``    → загружается state_dict из файла.
    - Одиночный файл ``.pt`` / ``.bin`` / ``.ckpt`` → загружается state_dict напрямую.

    Args:
        model: Базовая модель (до навешивания весов).
        ckpt_path: Путь к директории или файлу с весами.
        device: Устройство для загрузки (``'cpu'`` / ``'cuda'``). По умолчанию ``'cpu'``.

    Returns:
        Модель с загруженными весами — либо оригинальная с обновлённым
        state_dict, либо обёрнутая в PeftModel.

    Raises:
        FileNotFoundError: Если путь не существует или директория не содержит
            ни ``adapter_config.json``, ни ``pytorch_model.bin``.
        ImportError: Если требуется загрузка LoRA-адаптера, но peft не установлен.
    """
    path = Path(ckpt_path)
    logger.info("Загрузка весов из: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Указанный путь не существует: {path}")

    if path.is_dir():
        if (path / "adapter_config.json").exists():
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError("Для загрузки LoRA-адаптера необходима библиотека peft.") from e

            logger.info("Обнаружен LoRA адаптер — навешиваем на базовую модель.")
            return PeftModel.from_pretrained(model, str(path))

        weight_path = path / "pytorch_model.bin"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"В директории {path} не найден ни adapter_config.json, ни pytorch_model.bin"
            )
    else:
        weight_path = path

    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)

    # PL-чекпоинт оборачивает веса в ключ "state_dict" и добавляет префикс "model."
    if "state_dict" in checkpoint:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint

    result = model.load_state_dict(state_dict, strict=False)

    # strict=False глотает несоответствия без сигнала — логируем явно,
    # чтобы случайная опечатка в ключе не осталась незамеченной
    if result.missing_keys:
        logger.warning(
            "Отсутствующие ключи в чекпоинте (%d шт.): %s",
            len(result.missing_keys),
            result.missing_keys,
        )
    if result.unexpected_keys:
        logger.warning(
            "Неожиданные ключи в чекпоинте (%d шт.): %s",
            len(result.unexpected_keys),
            result.unexpected_keys,
        )
    if not result.missing_keys and not result.unexpected_keys:
        logger.info("Веса загружены без расхождений.")
    else:
        logger.info("Веса загружены (с расхождениями — см. предупреждения выше).")

    return model
