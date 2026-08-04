from unittest.mock import patch

import pytest
from omegaconf import ListConfig, OmegaConf

from src.utils.hydra_utils import _resolve_training_callbacks, setup_config


class TestHydraUtils:
    def test_resolve_callbacks_dict(self):
        """Словарь коллбэков должен конвертироваться в список (или ListConfig)."""
        cfg = OmegaConf.create(
            {
                "callbacks": {
                    "checkpoint": {"type": "ModelCheckpoint"},
                    "early_stopping": {"type": "EarlyStopping"},
                }
            }
        )
        res = _resolve_training_callbacks(cfg)

        # OmegaConf автоматически конвертирует list обратно в ListConfig
        assert isinstance(res.callbacks, (list, ListConfig))
        assert len(res.callbacks) == 2
        assert res.callbacks[0].type == "ModelCheckpoint"

    def test_resolve_callbacks_error_on_string(self):
        """Если Hydra смёржила строки вместо конфигов - кидаем ошибку."""
        cfg = OmegaConf.create(
            {
                "callbacks": ["checkpoint", "early_stopping"]  # Голоые строки
            }
        )
        with pytest.raises(ValueError, match="обнаружены строки вместо конфигов"):
            _resolve_training_callbacks(cfg)

    @patch("src.utils.hydra_utils.OmegaConf.merge")
    def test_setup_config_dynamic_pipelines(self, mock_merge):
        """Проверка динамического извлечения секции training."""
        cfg = OmegaConf.create(
            {
                "paths": {},
                "decoder_pipeline": {"training": {"callbacks": {}}},
                "some_new_pipeline": {"training": {"callbacks": {}}},
                "other_config": {"not_training": {}},
            }
        )

        # Эмулируем возврат валидированного конфига
        mock_merge.return_value = OmegaConf.create(
            {"paths": {}, "decoder_pipeline": {}, "some_new_pipeline": {}}
        )

        res = setup_config(cfg)

        # Оба пайплайна должны получить свой training обратно в виде списка
        assert hasattr(res.decoder_pipeline, "training")
        assert hasattr(res.some_new_pipeline, "training")
        assert isinstance(res.decoder_pipeline.training.callbacks, (list, ListConfig))
