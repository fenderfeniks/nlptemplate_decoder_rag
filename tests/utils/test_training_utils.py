from unittest.mock import patch

from omegaconf import OmegaConf

from src.utils.training_utils import resolve_resume_path


def make_cfg(resume_training=None, log_dir="/tmp/logs"):
    """Создаёт минимальный DictConfig для тестов."""
    d = {"paths": {"log_dir": log_dir}}
    if resume_training is not None:
        d["resume_training"] = resume_training
    return OmegaConf.create(d)


class TestResumeResumePath:
    # ------------------------------------------------------------------
    # resume_training=False / отсутствует
    # ------------------------------------------------------------------

    def test_returns_none_when_resume_false(self):
        """resume_training=False → None, чекпоинт не ищется."""
        cfg = make_cfg(resume_training=False)
        assert resolve_resume_path(cfg) is None

    def test_returns_none_when_resume_absent(self):
        """Ключ resume_training отсутствует → None."""
        cfg = make_cfg()  # resume_training не задан
        assert resolve_resume_path(cfg) is None

    # ------------------------------------------------------------------
    # resume_training=True, last.ckpt существует
    # ------------------------------------------------------------------

    def test_returns_path_when_ckpt_exists(self, tmp_path):
        """resume_training=True и last.ckpt существует → возвращаем путь."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        last_ckpt = ckpt_dir / "last.ckpt"
        last_ckpt.touch()

        cfg = make_cfg(resume_training=True, log_dir=str(tmp_path))
        result = resolve_resume_path(cfg)

        assert result == str(last_ckpt)

    def test_return_type_is_str(self, tmp_path):
        """Возвращаемое значение — строка, а не Path."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()

        cfg = make_cfg(resume_training=True, log_dir=str(tmp_path))
        result = resolve_resume_path(cfg)

        assert isinstance(result, str)

    # ------------------------------------------------------------------
    # resume_training=True, last.ckpt не существует
    # ------------------------------------------------------------------

    def test_returns_none_when_ckpt_missing(self, tmp_path):
        """resume_training=True, но last.ckpt нет → None (старт с нуля)."""
        cfg = make_cfg(resume_training=True, log_dir=str(tmp_path))
        result = resolve_resume_path(cfg)
        assert result is None

    @patch("src.utils.training_utils.logger")
    def test_warning_logged_when_ckpt_missing(self, mock_logger, tmp_path):
        """При отсутствии last.ckpt логируется предупреждение."""
        cfg = make_cfg(resume_training=True, log_dir=str(tmp_path))
        resolve_resume_path(cfg)

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "last.ckpt" in warning_msg or "resume_training" in warning_msg

    @patch("src.utils.training_utils.logger")
    def test_info_logged_when_ckpt_found(self, mock_logger, tmp_path):
        """При успешном обнаружении last.ckpt логируется info."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()

        cfg = make_cfg(resume_training=True, log_dir=str(tmp_path))
        resolve_resume_path(cfg)

        mock_logger.info.assert_called_once()
        info_msg = mock_logger.info.call_args[0][0]
        assert "Resume" in info_msg or "найден" in info_msg

    # ------------------------------------------------------------------
    # Проверка корректности пути
    # ------------------------------------------------------------------

    def test_path_contains_checkpoints_and_last(self, tmp_path):
        """Возвращаемый путь содержит .../checkpoints/last.ckpt."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()

        cfg = make_cfg(resume_training=True, log_dir=str(tmp_path))
        result = resolve_resume_path(cfg)

        assert result.endswith("checkpoints/last.ckpt") or result.endswith(r"checkpoints\last.ckpt")
