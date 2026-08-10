from unittest.mock import patch

from src.utils.mlflow.dependencies import (
    _strip_version_specifier,
    get_inference_pip_requirements,
)


class TestDependencies:
    def test_strip_version_specifier(self):
        """Проверка очистки имени пакета от версий и extras."""
        assert _strip_version_specifier("torch>=2.0.0") == "torch"
        assert _strip_version_specifier("transformers[torch]==4.40.0") == "transformers"
        assert _strip_version_specifier("numpy<2.0") == "numpy"
        assert _strip_version_specifier("sentencepiece") == "sentencepiece"

    @patch("src.utils.mlflow.dependencies.tomllib.load")
    @patch("src.utils.mlflow.dependencies.version")
    def test_get_inference_pip_requirements(self, mock_version, mock_toml_load, tmp_path):
        """Проверка успешного извлечения зависимостей."""
        # Фейковое содержимое pyproject.toml
        mock_toml_load.return_value = {
            "project": {
                "optional-dependencies": {"inference-core": ["torch>=2.0", "transformers[torch]"]}
            }
        }

        # Фейковые установленные версии
        def fake_version(pkg_name):
            return {"torch": "2.2.0", "transformers": "4.38.2"}[pkg_name]

        mock_version.side_effect = fake_version

        # Создаем пустой файл, чтобы open() не упал
        dummy_toml = tmp_path / "pyproject.toml"
        dummy_toml.touch()

        pinned = get_inference_pip_requirements(dummy_toml)

        assert pinned == ["torch==2.2.0", "transformers==4.38.2"]

    @patch("src.utils.mlflow.dependencies.tomllib.load")
    def test_get_inference_pip_requirements_missing_group(self, mock_toml_load, tmp_path):
        """Проверка, если секции inference-core нет в конфиге."""
        mock_toml_load.return_value = {"project": {"optional-dependencies": {}}}

        dummy_toml = tmp_path / "pyproject.toml"
        dummy_toml.touch()

        assert get_inference_pip_requirements(dummy_toml) == []
