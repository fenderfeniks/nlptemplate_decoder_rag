from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from src.tools.storage.resolver import ArtifactResolver


class TestArtifactResolver:
    def test_resolve_lora_load_type(self):
        """Проверка режима LoRA: должен пропатчить model_name_or_path и вернуть путь к адаптеру."""
        router = MagicMock()
        router.download_manifest.return_value = {
            "load_type": "lora",
            "base_model_uri": "hf://fenderfeniks/nlp_template_encoder",
            "lora_uri": "s3://adapters/prod_v1",
        }
        router.download_from_uri.return_value = Path("/cache/adapter")

        resolver = ArtifactResolver(router=router, cache_base_dir="/cache")
        cfg = OmegaConf.create(
            {"my_pipeline": {"model": {"builder": {}, "modifiers": {"finetuning": {}}}}}
        )

        db_dir, lora_path = resolver.resolve_and_patch(
            cfg=cfg, manifest_uri="local://manifest.json", pipeline_name="my_pipeline"
        )

        # В случае с HF префиксом 'hf://' должно срезаться
        assert (
            cfg.my_pipeline.model.builder.model_name_or_path == "fenderfeniks/nlp_template_encoder"
        )
        assert lora_path == Path("/cache/adapter")
        assert db_dir is None

    def test_resolve_full_model_load_type(self):
        """Проверка режима Full Model: должен проставить skip_peft=True."""
        router = MagicMock()
        router.download_manifest.return_value = {
            "load_type": "full_model",
            "model_uri": "s3://merged/prod_v2",
        }
        router.download_from_uri.return_value = Path("/cache/merged_model")

        resolver = ArtifactResolver(router=router, cache_base_dir="/cache")
        cfg = OmegaConf.create(
            {"qa_pipeline": {"model": {"builder": {}, "modifiers": {"finetuning": {}}}}}
        )

        db_dir, lora_path = resolver.resolve_and_patch(
            cfg=cfg, manifest_uri="local://manifest.json", pipeline_name="qa_pipeline"
        )

        assert cfg.qa_pipeline.model.builder.model_name_or_path == str(Path("/cache/merged_model"))
        assert cfg.qa_pipeline.model.modifiers.finetuning.skip_peft is True
        assert lora_path is None

    def test_resolve_vector_db_uri(self):
        """Для RAG должен возвращать путь к векторной БД, если она есть в манифесте."""
        router = MagicMock()
        router.download_manifest.return_value = {
            "load_type": "full_model",
            "model_uri": "s3://...",
            "vector_db_uri": "s3://vector_stores/v1",
        }
        router.download_from_uri.side_effect = [
            Path("/cache/vector_db"),
            Path("/cache/merged_model"),
        ]

        resolver = ArtifactResolver(router=router, cache_base_dir="/cache")
        cfg = OmegaConf.create({"rag": {"model": {"builder": {}, "modifiers": {"finetuning": {}}}}})

        db_dir, lora_path = resolver.resolve_and_patch(cfg, "local://manifest.json", "rag")

        assert db_dir == Path("/cache/vector_db")
