# tests/tools/test_data_tools.py
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from src.tools.batch_analytics import main as batch_analytics_main
from src.tools.fetch_data import fetch_data


class TestFetchDataTool:
    @patch("src.tools.fetch_data.setup_config")
    def test_fetch_data_creates_directory(self, mock_setup_config, tmp_path):
        """Проверяем, что скрипт корректно создает директорию для сырых данных."""
        raw_dir = tmp_path / "raw"

        cfg = OmegaConf.create(
            {
                "pipeline_name": "test_pipeline",
                "test_pipeline": {"data": {"paths": {"raw_data_dir": str(raw_dir)}}},
            }
        )
        mock_setup_config.return_value = cfg

        # Запускаем распакованную функцию (без Hydra декоратора)
        fetch_data.__wrapped__(cfg)

        assert raw_dir.exists()
        assert raw_dir.is_dir()


class TestBatchAnalyticsTool:
    @patch("src.tools.batch_analytics.hydra.utils.instantiate")
    @patch("src.tools.batch_analytics.setup_config")
    def test_batch_analytics_decoder_pipeline(self, mock_setup_config, mock_instantiate):
        """Проверка аналитики для decoder пайплайна (колонка generated_text)."""
        cfg = OmegaConf.create(
            {"pipeline_name": "decoder_pipeline", "decoder_pipeline": {"inference": {}}}
        )
        mock_setup_config.return_value = cfg

        mock_pipeline = MagicMock()
        # Имитируем возврат LLM (список словарей)
        mock_pipeline.return_value = [{"generated_text": "Ответ 1"}, {"generated_text": "Ответ 2"}]
        mock_instantiate.return_value = mock_pipeline

        batch_analytics_main.__wrapped__(cfg)
        mock_pipeline.assert_called_once()

    @patch("src.tools.batch_analytics.hydra.utils.instantiate")
    @patch("src.tools.batch_analytics.setup_config")
    def test_batch_analytics_rag_pipeline(self, mock_setup_config, mock_instantiate):
        """Проверка аналитики для rag пайплайна (колонка embedding)."""
        cfg = OmegaConf.create({"pipeline_name": "rag_pipeline", "rag_pipeline": {"inference": {}}})
        mock_setup_config.return_value = cfg

        mock_pipeline = MagicMock()
        # Имитируем возврат Embedder (список массивов)
        mock_pipeline.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_instantiate.return_value = mock_pipeline

        batch_analytics_main.__wrapped__(cfg)
        mock_pipeline.assert_called_once()
