# src/pipelines/decoder/training/post_training.py
"""Post-training утилиты для decoder-пайплайна."""

import logging

import torch

# Импорт протокола нужен только для type hinting
from src.utils.logging.protocol import ExperimentLogger
from src.utils.torch_utils import register_safe_globals


logger = logging.getLogger(__name__)


def run_post_training_evaluation(
    trainer, model_module, datamodule, experiment_logger: ExperimentLogger
):
    """Запускает финальную оценку на лучшем чекпоинте после обучения.

    Загружает лучшие веса LoRA из checkpoint_callback, затем запускает
    генерацию на test_dataset_raw через DecoderEvaluator.

    trainer.test() не используется — loss на тесте не нужен,
    бизнес-метрики (ROUGE, BLEU и т.д.) считаются через DecoderEvaluator.

    Returns:
        float | None: best_model_score из checkpoint_callback.
    """
    best_ckpt_path = trainer.checkpoint_callback.best_model_path

    if not best_ckpt_path:
        logger.warning("Лучший чекпоинт не найден — тест на текущих весах.")

    register_safe_globals()

    if best_ckpt_path:
        logger.info("Загрузка лучших весов из %s...", best_ckpt_path)
        checkpoint = torch.load(
            best_ckpt_path,
            map_location=model_module.device,
            weights_only=False,
        )
        lora_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}
        logger.info("LoRA тензоров найдено: %d.", len(lora_state_dict))
        model_module.load_state_dict(lora_state_dict, strict=False)

    # Инициализируем test_dataset_raw если ещё не было
    if datamodule.test_dataset_raw is None:
        datamodule.setup(stage="test")

    if datamodule.test_dataset_raw is None:
        logger.warning("test_dataset_raw недоступен — финальная генерация пропущена.")
        score = trainer.checkpoint_callback.best_model_score
        return float(score) if score is not None else None

    # Получаем run_id через интерфейс протокола
    run_id = experiment_logger.get_run_id(trainer)
    logger.info("Финальная оценка на тестовом бенчмарке (run_id=%s)...", run_id)

    # Ищем GenerationEvaluationCallback и запускаем evaluate напрямую
    evaluation_ran = False
    for callback in trainer.callbacks:
        if not (hasattr(callback, "_evaluator") and hasattr(callback, "_setup_eval_env")):
            continue

        callback._setup_eval_env(trainer, stage="test")

        if not callback._env_ready.get("test", False):
            logger.warning("test eval_dataset не готов — генерация пропущена.")
            continue

        # Используем контекстный менеджер из интерфейса логгера
        with experiment_logger.reopen_run(run_id) if run_id else _null_context():
            callback._evaluator.evaluate(
                stage="test",
                metrics_logger=experiment_logger,  # Передаем правильный логгер
                trainer=trainer,
                pl_module=model_module,
                global_step=trainer.global_step,
            )

        evaluation_ran = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not evaluation_ran:
        logger.warning(
            "GenerationEvaluationCallback не найден в trainer.callbacks — "
            "финальная генерация не запущена."
        )

    score = trainer.checkpoint_callback.best_model_score
    return float(score) if score is not None else None


class _null_context:  # noqa
    """Контекстный менеджер-заглушка когда run_id недоступен."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
