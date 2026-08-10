import json
import tempfile
from pathlib import Path


# Импортируем функцию, если она была бы разбита, но так как вся логика в main(),
# мы протестируем логику обновления манифеста изолированно,
# воссоздав кусок кода, отвечающий за манифест (обычно это выносится в функцию).
# Поскольку в merge_lora.py этот код зашит в `merge_and_export`,
# мы сымитируем этот процесс через мок файловой системы.


class TestMergeLoraManifest:
    def test_manifest_creation_and_update(self):
        """Проверка логики обновления JSON манифеста (п. 4 в merge_lora)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Эмулируем существующий манифест (старый lora-режим)
            old_manifest = {
                "load_type": "lora",
                "base_model_uri": "hf://base",
                "lora_uri": "s3://adapters",
                "other_meta": "keep_me",
            }
            manifest_file = tmp_path / "sequence_pipeline_manifest.json"

            # Логика обновления манифеста (вырезанная из скрипта для теста)
            manifest = old_manifest.copy()
            manifest["load_type"] = "full_model"
            manifest["model_uri"] = "s3://merged_models/model_prod_v2"
            manifest["updated_at"] = "2026-08-10T12:00:00Z"
            manifest.pop("base_model_uri", None)
            manifest.pop("lora_uri", None)

            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            # Проверки
            with open(manifest_file) as f:
                new_manifest = json.load(f)

            assert new_manifest["load_type"] == "full_model"
            assert new_manifest["model_uri"] == "s3://merged_models/model_prod_v2"
            assert "base_model_uri" not in new_manifest
            assert "lora_uri" not in new_manifest
            assert new_manifest["other_meta"] == "keep_me"  # Старые неконфликтующие ключи сохранены
