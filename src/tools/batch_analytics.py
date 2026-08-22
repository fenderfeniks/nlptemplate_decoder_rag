# src/tools/batch_analytics.py
"""Батч-аналитика для encoder-шаблона.

Поддерживаемые пайплайны и их входные/выходные колонки:

  sequence_pipeline  -> text_column -> predicted_label / predicted_score
  similarity_pipeline (encoding) -> encoding_column -> embedding
  token_pipeline     -> tokens_column -> predicted_tags
  qa_pipeline        -> question_column + context_column -> predicted_answer
"""

import logging
import sys

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Маппинг пайплайнов -> входные колонки и функция подготовки входа для пайплайна
# ---------------------------------------------------------------------------


def _build_sequence_input(df: pd.DataFrame, pipeline_cfg: DictConfig) -> list:
    """Возвращает список текстов для sequence_pipeline (multiclass / NLI / regression)."""
    data_cfg = pipeline_cfg.data
    text_col = data_cfg.get("text_column", "text")
    text_pair_col = data_cfg.get("text_pair_column", None)

    if text_pair_col and text_pair_col in df.columns:
        # NLI-режим: пары (premise, hypothesis)
        return list(zip(df[text_col].tolist(), df[text_pair_col].tolist(), strict=True))
    return df[text_col].tolist()


def _build_similarity_input(df: pd.DataFrame, pipeline_cfg: DictConfig) -> list:
    """Возвращает список текстов для similarity_pipeline (encoding / bi_encoder)."""
    data_cfg = pipeline_cfg.data
    encoding_col = data_cfg.get("encoding_column", "text")
    return df[encoding_col].tolist()


def _build_token_input(df: pd.DataFrame, pipeline_cfg: DictConfig) -> list:
    """Возвращает список токен-списков для token_pipeline."""
    data_cfg = pipeline_cfg.data
    tokens_col = data_cfg.get("tokens_column", "tokens")
    return df[tokens_col].tolist()


def _build_qa_input(df: pd.DataFrame, pipeline_cfg: DictConfig) -> list:
    """Возвращает список пар (question, context) для qa_pipeline."""
    data_cfg = pipeline_cfg.data
    question_col = data_cfg.get("question_column", "question")
    context_col = data_cfg.get("context_column", "context")
    return list(zip(df[question_col].tolist(), df[context_col].tolist(), strict=True))


# ---------------------------------------------------------------------------
# Моковые данные на случай отсутствия источника
# ---------------------------------------------------------------------------


def _build_mock_df(pipeline_name: str, pipeline_cfg: DictConfig) -> pd.DataFrame:
    """Возвращает минимальный DataFrame с нужными колонками для теста."""
    data_cfg = pipeline_cfg.data

    if "sequence" in pipeline_name:
        text_col = data_cfg.get("text_column", "text")
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                text_col: [
                    "Компания показала рекордную выручку в третьем квартале.",
                    "Новый препарат прошёл успешные клинические испытания.",
                    "Футбольная команда вышла в финал чемпионата.",
                ],
            }
        )

    if "similarity" in pipeline_name:
        encoding_col = data_cfg.get("encoding_column", "text")
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                encoding_col: [
                    "Как работает градиентный спуск?",
                    "Что такое трансформерная архитектура?",
                    "Объясни механизм внимания в нейронных сетях.",
                ],
            }
        )

    if "token" in pipeline_name:
        tokens_col = data_cfg.get("tokens_column", "tokens")
        return pd.DataFrame(
            {
                "id": [1, 2],
                tokens_col: [
                    ["Иван", "Петров", "работает", "в", "Москве"],
                    ["Google", "объявила", "о", "новом", "продукте"],
                ],
            }
        )

    if "qa" in pipeline_name:
        question_col = data_cfg.get("question_column", "question")
        context_col = data_cfg.get("context_column", "context")
        return pd.DataFrame(
            {
                "id": [1, 2],
                question_col: [
                    "Когда была основана компания?",
                    "Кто является CEO организации?",
                ],
                context_col: [
                    "Компания была основана в 1998 году двумя студентами Стэнфорда.",
                    "Организацию возглавляет Джон Смит, занявший пост CEO в 2015 году.",
                ],
            }
        )

    raise ValueError(
        f"Неизвестный pipeline_name: '{pipeline_name}'. "
        f"Ожидается одно из: sequence_pipeline, similarity_pipeline, token_pipeline, qa_pipeline."
    )


# ---------------------------------------------------------------------------
# Маппинг: pipeline_name -> (input_builder, output_column)
# ---------------------------------------------------------------------------

_PIPELINE_CONFIG = {
    "sequence_pipeline": (_build_sequence_input, "predicted_label"),
    "similarity_pipeline": (_build_similarity_input, "embedding"),
    "token_pipeline": (_build_token_input, "predicted_tags"),
    "qa_pipeline": (_build_qa_input, "predicted_answer"),
}


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Батч-аналитика с динамической инициализацией инференс-пайплайна."""
    cfg = setup_config(cfg)

    pipeline_name = cfg.pipeline_name
    pipeline_cfg = getattr(cfg, pipeline_name)

    logger.info("Запуск батч-аналитики для пайплайна: %s", pipeline_name)

    # ── 1. Определяем конфигурацию для текущего пайплайна ─────────────────
    matched_key = next(
        (key for key in _PIPELINE_CONFIG if pipeline_name.startswith(key.split("_")[0])),
        None,
    )
    # Точный матч приоритетнее префиксного
    if pipeline_name in _PIPELINE_CONFIG:
        matched_key = pipeline_name

    if matched_key is None:
        logger.error(
            "Пайплайн '%s' не поддерживается batch_analytics. Доступные: %s",
            pipeline_name,
            list(_PIPELINE_CONFIG.keys()),
        )
        sys.exit(1)

    input_builder, output_column = _PIPELINE_CONFIG[matched_key]

    # ── 2. Инициализация инференс-пайплайна ───────────────────────────────
    try:
        logger.info("Сборка инференс-пайплайна через Hydra...")
        pipeline = hydra.utils.instantiate(pipeline_cfg.inference, cfg=cfg)
    except Exception as e:
        logger.exception("Не удалось инициализировать пайплайн инференса: %s", e)
        sys.exit(1)

    # ── 3. Подготовка данных ───────────────────────────────────────────────
    # TODO: заменить моковые данные на реальный источник (БД, S3, CSV и т.д.)
    df = _build_mock_df(pipeline_name, pipeline_cfg)
    logger.info("Данные подготовлены: %d записей, колонки: %s", len(df), list(df.columns))

    # ── 4. Формируем вход в формате, который ожидает пайплайн ─────────────
    inputs = input_builder(df, pipeline_cfg)
    logger.info("Запуск батч-инференса (%d записей)...", len(inputs))

    # ── 5. Инференс ───────────────────────────────────────────────────────
    results = pipeline(inputs)

    # ── 6. Сохраняем результаты в DataFrame ───────────────────────────────
    if isinstance(results, list) and len(results) == len(df):
        df[output_column] = results
    else:
        # Пайплайн вернул dict с несколькими полями (например, QA: start + end + text)
        if isinstance(results, list) and isinstance(results[0], dict):
            for key in results[0]:
                df[f"predicted_{key}"] = [r[key] for r in results]
        else:
            logger.warning("Неожиданный формат результатов: %s. Сохраняем как есть.", type(results))
            df[output_column] = results

    logger.info("Пример результатов:\n%s", df.head().to_string())
    logger.info("Батч-аналитика успешно завершена.")


if __name__ == "__main__":
    main()
