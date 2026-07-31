import os
import time
from unittest.mock import patch

from src.tools.maintenance import cleanup_mlruns


class TestMaintenanceCleanup:
    def test_cleanup_removes_old_files_and_empty_dirs(self, tmp_path):
        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()

        fresh_file = mlruns_dir / "fresh.txt"
        fresh_file.write_text("fresh")

        # 1. Папка с файлом внутри
        old_dir_with_file = mlruns_dir / "old_run"
        old_dir_with_file.mkdir()
        old_file = old_dir_with_file / "old.txt"
        old_file.write_text("old")

        # 2. Изначально пустая старая папка
        empty_old_dir = mlruns_dir / "empty_old_dir"
        empty_old_dir.mkdir()

        forty_days_ago = time.time() - (40 * 24 * 60 * 60)
        os.utime(str(old_file), (forty_days_ago, forty_days_ago))
        os.utime(str(old_dir_with_file), (forty_days_ago, forty_days_ago))
        os.utime(str(empty_old_dir), (forty_days_ago, forty_days_ago))

        with patch("src.tools.maintenance.os.getenv", return_value=str(mlruns_dir)):
            cleanup_mlruns(days=30)

        assert fresh_file.exists()
        assert not old_file.exists()
        # Изначально пустая старая папка удаляется гарантированно
        assert not empty_old_dir.exists()
        # old_dir_with_file мы не ассертим: на Windows её mtime обновился при удалении файла,
        # поэтому скрипт честно оставил её до следующего запуска.

    def test_cleanup_skips_nonexistent_dir(self, caplog, tmp_path):
        fake_dir = tmp_path / "does_not_exist"

        with patch("src.tools.maintenance.os.getenv", return_value=str(fake_dir)):
            cleanup_mlruns(days=30)

        assert "не существует. Очистка пропущена" in caplog.text
