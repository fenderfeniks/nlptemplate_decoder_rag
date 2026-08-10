import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.utils.notebook_setup import _ensure_on_path, _find_project_root, _init_hydra


class TestFindProjectRoot:
    @patch("src.utils.notebook_setup.Path.cwd")
    def test_find_project_root_success(self, mock_cwd, tmp_path):
        """Ищет pyproject.toml поднимаясь по дереву директорий."""
        root = tmp_path
        (root / "pyproject.toml").touch()

        sub2 = root / "sub1" / "sub2"
        sub2.mkdir(parents=True)

        mock_cwd.return_value = sub2

        assert _find_project_root() == root

    @patch("src.utils.notebook_setup.Path.cwd")
    def test_find_project_root_fails(self, mock_cwd, tmp_path):
        """Если pyproject.toml нигде не найден — выбрасывается FileNotFoundError."""
        mock_cwd.return_value = tmp_path

        with pytest.raises(FileNotFoundError, match="Could not locate project root"):
            _find_project_root()

    @patch("src.utils.notebook_setup.Path.cwd")
    def test_find_project_root_in_cwd(self, mock_cwd, tmp_path):
        """pyproject.toml прямо в cwd — возвращает cwd."""
        (tmp_path / "pyproject.toml").touch()
        mock_cwd.return_value = tmp_path

        assert _find_project_root() == tmp_path

    @patch("src.utils.notebook_setup.Path.cwd")
    def test_find_project_root_at_second_level(self, mock_cwd, tmp_path):
        """pyproject.toml на один уровень выше cwd."""
        (tmp_path / "pyproject.toml").touch()
        child = tmp_path / "subdir"
        child.mkdir()
        mock_cwd.return_value = child

        assert _find_project_root() == tmp_path


class TestEnsureOnPath:
    def test_adds_directory_to_sys_path(self, tmp_path):
        """Путь добавляется в sys.path, если его там нет."""
        path_str = str(tmp_path / "new_dir")
        # Убеждаемся, что его там нет
        if path_str in sys.path:
            sys.path.remove(path_str)

        _ensure_on_path(Path(path_str))

        assert path_str in sys.path
        sys.path.remove(path_str)  # чистим после теста

    def test_does_not_duplicate_existing_path(self, tmp_path):
        """Если путь уже в sys.path — не добавляется дважды."""
        path_str = str(tmp_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

        before_count = sys.path.count(path_str)
        _ensure_on_path(tmp_path)
        after_count = sys.path.count(path_str)

        assert before_count == after_count


class TestInitHydra:
    def test_init_hydra_calls_initialize_and_compose(self, tmp_path):
        """_init_hydra очищает глобальный Hydra, инициализирует и вызывает compose."""
        mock_cfg = MagicMock()

        with (
            patch("src.utils.notebook_setup.GlobalHydra") as mock_global_hydra,
            patch("src.utils.notebook_setup.initialize_config_dir") as mock_init,
            patch("src.utils.notebook_setup.compose", return_value=mock_cfg) as mock_compose,
        ):
            result = _init_hydra(config_dir=str(tmp_path), config_name="main")

        mock_global_hydra.instance().clear.assert_called_once()
        mock_init.assert_called_once_with(config_dir=str(tmp_path), version_base="1.3")
        mock_compose.assert_called_once_with(config_name="main")
        assert result is mock_cfg


class TestSetupNotebook:
    @patch("src.utils.notebook_setup._find_project_root")
    @patch("src.utils.notebook_setup._ensure_on_path")
    @patch("src.utils.notebook_setup._init_hydra")
    def test_setup_notebook_wires_components(
        self, mock_init_hydra, mock_ensure_on_path, mock_find_root, tmp_path
    ):
        """setup_notebook находит корень, добавляет в путь и инициализирует Hydra."""
        from src.utils.notebook_setup import setup_notebook

        mock_find_root.return_value = tmp_path
        mock_cfg = MagicMock()
        mock_init_hydra.return_value = mock_cfg

        result = setup_notebook(config_name="main")

        mock_find_root.assert_called_once()
        mock_ensure_on_path.assert_called_once_with(tmp_path)
        mock_init_hydra.assert_called_once_with(
            config_dir=str(tmp_path / "configs"),
            config_name="main",
        )
        assert result is mock_cfg

    @patch("src.utils.notebook_setup._find_project_root")
    @patch("src.utils.notebook_setup._ensure_on_path")
    @patch("src.utils.notebook_setup._init_hydra")
    def test_setup_notebook_custom_config_name(
        self, mock_init_hydra, mock_ensure_on_path, mock_find_root, tmp_path
    ):
        """config_name пробрасывается в _init_hydra."""
        from src.utils.notebook_setup import setup_notebook

        mock_find_root.return_value = tmp_path
        mock_init_hydra.return_value = MagicMock()

        setup_notebook(config_name="experiment")

        mock_init_hydra.assert_called_once_with(
            config_dir=str(tmp_path / "configs"),
            config_name="experiment",
        )
