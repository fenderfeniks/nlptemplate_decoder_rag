# scripts/decoder/eval.py
"""Оценка качества декодер-модели на зафиксированной эталонной выборке."""

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig


load_dotenv()

from src.endpoints.eval import EvalContext, run_universal_eval  # noqa: E402
from src.evaluation.evaluators.decoder import DecoderEvaluator  # noqa: E402
from src.pipelines.decoder.inference.builder import build_decoder_model  # noqa: E402
from src.utils.hydra_utils import setup_config  # noqa: E402
from src.utils.logger import setup_logging  # noqa: E402


setup_logging()
logger = logging.getLogger(__name__)


def _build_and_eval(ctx: EvalContext) -> dict[str, float]:
    cfg = ctx.cfg

    # Инициализируем PromptManager из конфига (те же шаблоны что при обучении)
    prompt_manager = hydra.utils.instantiate(cfg.data.transforms.prompt_formatting.prompt_manager)
    template_name = cfg.data.transforms.prompt_formatting.template_name
    retrieve_col = cfg.data.get("retrieve_column", None)

    # Форматируем промпты через тот же шаблон что используется при обучении
    formatted_queries = []
    for i, item in enumerate(ctx.benchmark_dataset):
        raw_prompt = ctx.queries[i]

        # Собираем kwargs — все колонки записи
        kwargs = dict(item)
        kwargs["instruction"] = raw_prompt  # на случай если колонка называется иначе

        # Контекст — None если колонки нет или она пустая
        if retrieve_col and retrieve_col in ctx.benchmark_dataset.column_names:
            raw_ctx = item.get(retrieve_col, None)
            kwargs["context"] = raw_ctx if raw_ctx else None
        else:
            kwargs["context"] = None

        formatted = prompt_manager.render(template_name, **kwargs)
        formatted_queries.append(formatted)

    eval_dataset = [
        {"prompt": q, "response": gt}
        for q, gt in zip(formatted_queries, ctx.ground_truths, strict=True)
    ]
    logger.info("Эталонный датасет сформирован (%d записей).", len(eval_dataset))

    # Остальное без изменений
    base_model, tokenizer = build_decoder_model(cfg, ctx.lora_path)
    evaluator = DecoderEvaluator(
        model_name=cfg.model.architecture.mlflow_model_name,
        generation_batch_size=cfg.get("generation_batch_size", 2),
        generation_kwargs=cfg.get("evaluation", {}).get("generation_kwargs", {}),
        metrics_cfg=cfg.get("evaluation", {}).get("metrics", None),
        num_random=cfg.get("evaluation", {}).get("num_random", len(eval_dataset)),
    )

    with ctx.experiment_logger.start_run(run_name="decoder_standalone_eval"):
        metrics = evaluator.evaluate(
            stage="test",
            metrics_logger=ctx.experiment_logger,
            model=base_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
        )

    return metrics


@hydra.main(config_path="../../configs", config_name="eval_decoder", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("Инициализация независимой оценки Decoder...")

    eval_cfg = cfg.get("eval", {})

    run_universal_eval(
        cfg=cfg,
        pipeline_name=cfg.get("pipeline_name", "decoder_pipeline"),
        build_and_eval_fn=_build_and_eval,
        query_column=eval_cfg.get("prompt_column", "prompt"),
        answer_column=eval_cfg.get("target_column", "response"),
        cache_subdir="decoder_cache",
        require_db=False,
    )


if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline

    enforce_pipeline("decoder_pipeline")
    evaluate()
