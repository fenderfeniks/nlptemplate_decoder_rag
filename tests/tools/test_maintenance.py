import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from src.tools.maintenance import cleanup_mlruns


class TestMaintenance:
    def test_cleanup_mlruns_deletes_old_files(self):
        """Проверка удаления файлов старше X дней и сохранения свежих файлов."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MLRUNS_DIR": tmpdir}):
                tmp_path = Path(tmpdir)

                sub_dir = tmp_path / "run_123"
                sub_dir.mkdir()

                old_file = sub_dir / "old_checkpoint.ckpt"
                new_file = sub_dir / "new_events.out"

                old_file.touch()
                new_file.touch()

                forty_days_ago = time.time() - (40 * 24 * 60 * 60)
                os.utime(old_file, (forty_days_ago, forty_days_ago))

                cleanup_mlruns(days=30)

                assert not old_file.exists(), "Старый файл должен быть удален"
                assert new_file.exists(), "Свежий файл должен остаться"
                assert sub_dir.exists(), (
                    "Папка не должна удаляться, так как внутри есть свежий файл"
                )

    def test_cleanup_mlruns_removes_empty_dirs(self):
        """Проверка того, что папка удаляется, когда все файлы внутри нее устарели."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MLRUNS_DIR": tmpdir}):
                tmp_path = Path(tmpdir)
                sub_dir = tmp_path / "old_run"
                sub_dir.mkdir()

                old_file = sub_dir / "old.txt"
                old_file.touch()

                forty_days_ago = time.time() - (40 * 24 * 60 * 60)
                os.utime(old_file, (forty_days_ago, forty_days_ago))

                cleanup_mlruns(days=30)

                assert not old_file.exists(), "Старый файл должен быть удален"
                assert not sub_dir.exists(), "Пустая папка должна быть удалена"
