import asyncio
import json
import logging
import time
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.endpoints.infer import run_universal_infer
from src.pipelines.decoder.inference.builder import build_decoder_model
from src.pipelines.decoder.inference.generator import HFTextGenerator
from src.pipelines.decoder.inference.inference import LLMGenerationClient
from src.tools.storage.resolver import ArtifactResolver
from src.utils.cli import enforce_pipeline
from src.utils.hydra_utils import setup_config

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Вспомогательные функции генерации (остаются здесь)
# ---------------------------------------------------------------------------
def _call_generate(generator, queries: list[str]) -> list[str]:
    if isinstance(generator, LLMGenerationClient):
        return asyncio.run(generator.generate(queries))
    return generator.generate(queries)

def _measure(generator, tokenizer, queries: list[str]) -> tuple[list[str], int, float]:
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
            for q, gen in zip(queries, generated_texts)  # noqa: B905
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
        query, gen_text, gen_tokens, tps,
    )

# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------
def run_decoder_logic(cfg: DictConfig, resolver: ArtifactResolver) -> None:
    """Специфичная логика сборки и инференса Декодера."""
    _, lora_path, *_ = resolver.resolve_and_patch(
        cfg, cfg.system.manifest.uri, pipeline_name="decoder_pipeline", is_training=False
    )
    
    base_model, tokenizer = build_decoder_model(cfg, lora_path)
    generator = hydra.utils.instantiate(cfg.inference.generator, model=base_model, tokenizer=tokenizer)

    inference_cfg = cfg.inference
    input_file = inference_cfg.get("input_file")
    output_file = inference_cfg.get("output_file", "predictions.jsonl")
    query = str(cfg.get("text", "Объясни, что такое Retrieval-Augmented Generation (RAG)."))

    if input_file and Path(input_file).exists():
        _run_batch(generator, tokenizer, input_file, output_file)
    else:
        _run_single(generator, tokenizer, query)

@hydra.main(config_path="../../configs", config_name="eval_decoder", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    run_universal_infer(cfg, "decoder_pipeline", run_decoder_logic)

if __name__ == "__main__":
    enforce_pipeline("decoder_pipeline")
    main()