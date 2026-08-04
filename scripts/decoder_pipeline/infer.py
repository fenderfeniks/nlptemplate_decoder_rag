import asyncio
import gc
import json
import logging
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.pipelines.decoder.inference.generator import HFTextGenerator
from src.pipelines.decoder.inference.inference import LLMGenerationClient
from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("Инициализация decoder-пайплайна...")

    # 1. Резолвинг артефактов (веса базовой модели + LoRA-адаптер)
    router = hydra.utils.instantiate(cfg.storage_router)
    cache_base = Path(cfg.paths.model_dir) / "decoder_cache"
    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    manifest_uri = cfg.get("manifest_uri", "local://./prod_storage/manifests/decoder_manifest.json")

    try:
        resolver.resolve_and_patch(cfg, manifest_uri, pipeline_name="decoder_pipeline")
    except Exception as e:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Сбой подготовки артефактов decoder: %s", e)
        sys.exit(1)

    # 2. Сборка токенизатора и модели (пути уже пропатчены резолвером)
    tokenizer = hydra.utils.instantiate(cfg.decoder_pipeline.model.tokenizer).build()

    builder = hydra.utils.instantiate(cfg.decoder_pipeline.model.builder)
    builder.modifiers_cfg = cfg.decoder_pipeline.model.get("modifiers")
    model = builder.build(tokenizer=tokenizer)

    # 3. Сборка генератора
    generator = hydra.utils.instantiate(
        cfg.decoder_pipeline.inference,
        model=model,
        tokenizer=tokenizer,
    )

    _free_memory()

    # 4. Пакетная или одиночная генерация
    input_file = cfg.decoder_pipeline.inference.get("input_file")
    output_file = cfg.decoder_pipeline.inference.get("output_file", "predictions.jsonl")
    query = cfg.get("text", "Объясни, что такое Retrieval-Augmented Generation (RAG).")

    if input_file and Path(input_file).exists():
        _run_batch(generator, tokenizer, input_file, output_file)
    else:
        _run_single(generator, tokenizer, query)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _call_generate(generator, queries: list[str]) -> list[str]:
    """Единая точка вызова generate() для HFTextGenerator и LLMGenerationClient.

    HFTextGenerator.generate — синхронный.
    LLMGenerationClient.generate — async, запускаем через asyncio.run().
    """
    if isinstance(generator, LLMGenerationClient):
        return asyncio.run(generator.generate(queries))
    return generator.generate(queries)


def _measure(generator, tokenizer, queries: list[str]) -> tuple[list[str], int, float]:
    """Генерирует тексты и возвращает (тексты, суммарные токены, токенов/сек).

    Подсчёт токенов через tokenizer доступен только для HFTextGenerator.
    LLMGenerationClient токенизатора не имеет — считаем по словам как приближение.
    """
    start = time.perf_counter()
    texts = _call_generate(generator, queries)
    elapsed = time.perf_counter() - start

    if isinstance(generator, HFTextGenerator):
        total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    else:
        total_tokens = sum(len(t.split()) for t in texts)

    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    return texts, total_tokens, tps


def _run_batch(generator, tokenizer, input_file: str, output_file: str) -> None:
    logger.info("Запуск пакетного инференса из файла: %s", input_file)

    with open(input_file, encoding="utf-8") as f:
        queries = [json.loads(line)["prompt"] for line in f if line.strip()]

    generated_texts, _, tps = _measure(generator, tokenizer, queries)
    logger.info("Пакетная генерация завершена. Скорость: %.2f токенов/сек.", tps)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps({"prompt": q, "generated": gen}, ensure_ascii=False) + "\n"
            for q, gen in zip(queries, generated_texts)  # noqa
        )
    logger.info("Результаты сохранены в %s", output_file)


def _run_single(generator, tokenizer, query: str) -> None:
    logger.info("Запуск одиночной генерации...")

    generated_texts, gen_tokens, tps = _measure(generator, tokenizer, [query])
    gen_text = generated_texts[0]

    logger.info(
        "\n==================================================\n"
        "ПРОМПТ:\n%s\n"
        "--------------------------------------------------\n"
        "ОТВЕТ МОДЕЛИ:\n%s\n"
        "--------------------------------------------------\n"
        "ТЕЛЕМЕТРИЯ: %d токенов | Скорость: %.2f t/s\n"
        "==================================================",
        query,
        gen_text,
        gen_tokens,
        tps,
    )


if __name__ == "__main__":
    infer()
