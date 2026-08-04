from unittest.mock import patch

import pytest

from src.utils.notebook_setup import _find_project_root


class TestNotebookSetup:
    @patch("src.utils.notebook_setup.Path.cwd")
    def test_find_project_root_success(self, mock_cwd, tmp_path):
        """Ищет pyproject.toml поднимаясь по дереву директорий[cite: 35]."""
        # Создаем структуру: root/sub1/sub2
        root = tmp_path
        (root / "pyproject.toml").touch()

        sub2 = root / "sub1" / "sub2"
        sub2.mkdir(parents=True)

        mock_cwd.return_value = sub2

        assert _find_project_root() == root

    @patch("src.utils.notebook_setup.Path.cwd")
    def test_find_project_root_fails(self, mock_cwd, tmp_path):
        """Если файла нет, выбрасывается FileNotFoundError[cite: 35]."""
        mock_cwd.return_value = tmp_path

        with pytest.raises(FileNotFoundError, match="Could not locate project root"):
            _find_project_root()
