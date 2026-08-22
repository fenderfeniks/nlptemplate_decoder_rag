"""Построение эталонного бенчмарка для RAG-пайплайна."""

import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
import hydra  # noqa
from omegaconf import DictConfig  # noqa

from src.tools.benchmark.generator import LocalQAGenerator  # noqa
from src.tools.evaluation.judges.nli_judge import NLIJudge  # noqa
from src.tools.evaluation.schema import EvalInput  # noqa
from src.utils.hydra_utils import setup_config  # noqa
from src.utils.logger import setup_logging  # noqa


setup_logging()
logger = logging.getLogger(__name__)

BENCHMARK_FILENAME = "benchmark.jsonl"


def compute_chunk_id(text: str) -> str:
    """Вычисляет детерминированный ID чанка (идентично логике векторной БД)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@hydra.main(config_path="../../configs", config_name="build_benchmark_rag", version_base="1.3")
def build_benchmark(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    logger.info("=== Старт создания бенчмарка (RAG Pipeline) ===")

    storage_client = hydra.utils.instantiate(cfg.system.storage)
    router = hydra.utils.instantiate(cfg.system.storage_router)
    uri_prefix = getattr(cfg.system.storage, "uri_prefix", "")
    manifest_uri = cfg.system.manifest.uri

    # 1. Загрузка данных и динамическое применение чанкинга
    logger.info("Загрузка исходного датасета...")
    source = hydra.utils.instantiate(cfg.data.source)
    raw_datasets = source.load()
    base_ds = raw_datasets["train"] if "train" in raw_datasets else raw_datasets

    logger.info("Нарезка текстов на чанки через конфигурацию трансформации...")
    chunking_transform = hydra.utils.instantiate(cfg.data.transforms.chunking)
    chunked_ds = chunking_transform(base_ds)

    # Перемешивание, чтобы не собирать вопросы только по первой статье
    seed = cfg.get("seed", 42)
    shuffled_ds = chunked_ds.shuffle(seed=seed)

    # 2. Инициализация ML-моделей (Generator + NLIJudge)
    cache_base = Path(getattr(cfg.system, "cache_dir", ".cache"))
    cache_base.mkdir(parents=True, exist_ok=True)

    gen_type = cfg.benchmark.generator_type
    if gen_type == "local":
        logger.info("Инициализация LocalQAGenerator через манифест...")
        generator = LocalQAGenerator.from_manifest(
            router=router,
            manifest_uri=manifest_uri,
            cache_base=cache_base,
            gen_cfg=cfg.get("evaluation", {}).get("benchmark", {}).get("generator", {}),
        )
    else:
        logger.info("Инициализация APIQAGenerator...")
        generator = hydra.utils.instantiate(cfg.evaluation.benchmark.generator)

    logger.info("Инициализация NLIJudge для фильтрации галлюцинаций...")
    nli_judge = NLIJudge.from_manifest(
        router=router,
        manifest_uri=manifest_uri,
        cache_base=cache_base,
        verdict_threshold=cfg.benchmark.nli_threshold,
    )

    # 3. Основной цикл: Генерация и фильтрация
    benchmark_size = cfg.benchmark.benchmark_size
    text_column = cfg.benchmark.text_column
    min_length = cfg.benchmark.min_chunk_length

    records = []
    logger.info("Запуск сборки бенчмарка (цель: %d валидных пар)...", benchmark_size)

    for item in shuffled_ds:
        if len(records) >= benchmark_size:
            break

        chunk_text = str(item.get(text_column, "")).strip()
        if len(chunk_text) < min_length:
            continue

        chunk_id = compute_chunk_id(chunk_text)

        # Генерация (LLM/API)
        qa_pair = generator.generate(chunk_text)
        if not qa_pair:
            continue

        question, answer = qa_pair

        # Фильтрация (NLI)
        # premise = chunk_text (reference), hypothesis = answer (response)
        eval_inp = EvalInput(
            prompt=question, response=answer, reference=chunk_text, metadata={"chunk_id": chunk_id}
        )
        eval_result = nli_judge.evaluate_batch([eval_inp])[0]

        if eval_result.verdict:
            records.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "question": question,
                    "answer": answer,
                    "nli_score": eval_result.score,
                }
            )
            if len(records) % 10 == 0:
                logger.info("Собрано %d / %d пар...", len(records), benchmark_size)

    # 4. Выгрузка в Storage
    remote_benchmark_dir = f"{cfg.pipeline_name}/benchmarks/latest"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        benchmark_file = tmp_path / BENCHMARK_FILENAME

        with open(benchmark_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Выгрузка бенчмарка в Storage: %s", remote_benchmark_dir)
        storage_client.upload(local_dir=tmp_dir, remote_path=remote_benchmark_dir)

    # 5. Обновление единого манифеста
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            global_manifest = router.download_manifest(manifest_uri, cache_dir=tmp_path)
            logger.info("Обновляем секцию манифеста %s", cfg.pipeline_name)
        except Exception:
            logger.warning("Существующий манифест не найден. Будет создан новый.")
            global_manifest = {}

        if cfg.pipeline_name not in global_manifest:
            global_manifest[cfg.pipeline_name] = {}

        global_manifest[cfg.pipeline_name]["benchmark_uri"] = (
            f"{uri_prefix}{remote_benchmark_dir}/{BENCHMARK_FILENAME}"
        )
        global_manifest[cfg.pipeline_name]["benchmark_updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        global_manifest[cfg.pipeline_name]["benchmark_size"] = len(records)

        manifest_file = tmp_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(global_manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload_file(local_path=manifest_file, remote_path="manifest.json")

    logger.info(
        "=== RAG Бенчмарк успешно зафиксирован. URI: %s%s/%s ===",
        uri_prefix,
        remote_benchmark_dir,
        BENCHMARK_FILENAME,
    )


if __name__ == "__main__":
    from src.utils.cli import enforce_pipeline

    enforce_pipeline("rag_pipeline")
    build_benchmark()
