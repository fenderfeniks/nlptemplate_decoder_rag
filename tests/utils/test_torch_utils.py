from unittest.mock import patch

from src.utils.torch_utils import register_safe_globals


class TestTorchUtils:
    @patch("src.utils.torch_utils.torch.serialization.add_safe_globals")
    def test_register_safe_globals(self, mock_add_safe_globals):
        """Проверка регистрации безопасных объектов PyTorch."""
        register_safe_globals()

        # Убеждаемся, что функция была вызвана 1 раз и ей был передан список
        mock_add_safe_globals.assert_called_once()
        args = mock_add_safe_globals.call_args[0][0]

        assert isinstance(args, list)
        assert len(args) >= 2  # Как минимум functools.partial и AdamW из BASE_SAFE_GLOBALS
