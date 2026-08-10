import sys
from unittest.mock import patch

from src.utils.cli import enforce_pipeline


# ---------------------------------------------------------------------------
# Хелпер: запускаем функцию с изолированным sys.argv
# ---------------------------------------------------------------------------


def run_enforce(argv_tail: list[str], expected: str, *extra_overrides: str) -> list[str]:
    """Запускает enforce_pipeline с заданным хвостом argv и возвращает итоговый sys.argv."""
    with patch.object(sys, "argv", ["script.py"] + argv_tail):
        enforce_pipeline(expected, *extra_overrides)
        return list(sys.argv)


# ---------------------------------------------------------------------------
# pipeline_name — базовые сценарии
# ---------------------------------------------------------------------------


class TestEnforcePipeline:
    def test_adds_pipeline_name_when_absent(self):
        """pipeline_name отсутствует — добавляем его в конец."""
        result = run_enforce([], "rag_pipeline")
        assert "pipeline_name=rag_pipeline" in result

    def test_keeps_correct_pipeline_name(self):
        """pipeline_name уже корректный — оставляем как есть."""
        result = run_enforce(["pipeline_name=rag_pipeline"], "rag_pipeline")
        # Должен быть ровно один экземпляр
        assert result.count("pipeline_name=rag_pipeline") == 1

    def test_replaces_wrong_pipeline_name(self):
        """pipeline_name задан неверно — принудительно заменяем."""
        result = run_enforce(["pipeline_name=decoder_pipeline"], "rag_pipeline")
        assert "pipeline_name=rag_pipeline" in result
        assert "pipeline_name=decoder_pipeline" not in result

    def test_only_one_pipeline_name_entry(self):
        """В итоговом argv ровно один аргумент pipeline_name=."""
        result = run_enforce(["pipeline_name=other"], "rag_pipeline")
        matches = [a for a in result if a.startswith("pipeline_name=")]
        assert len(matches) == 1

    # ---------------------------------------------------------------------------
    # extra_overrides
    # ---------------------------------------------------------------------------

    def test_adds_extra_override_when_absent(self):
        """Дополнительный оверрайд добавляется, если его нет."""
        result = run_enforce([], "rag_pipeline", "rag_pipeline/data=indexing")
        assert "rag_pipeline/data=indexing" in result

    def test_does_not_duplicate_existing_override(self):
        """Дополнительный оверрайд не дублируется, если уже есть."""
        result = run_enforce(
            ["rag_pipeline/data=indexing"],
            "rag_pipeline",
            "rag_pipeline/data=indexing",
        )
        assert result.count("rag_pipeline/data=indexing") == 1

    def test_existing_override_with_different_value_not_replaced(self):
        """Если ключ уже есть с другим значением — не перезаписываем (проверяем по префиксу)."""
        result = run_enforce(
            ["rag_pipeline/data=query"],
            "rag_pipeline",
            "rag_pipeline/data=indexing",
        )
        # Уже существует rag_pipeline/data=query, новый не должен добавиться
        assert "rag_pipeline/data=indexing" not in result
        assert "rag_pipeline/data=query" in result

    def test_multiple_extra_overrides_added(self):
        """Несколько дополнительных оверрайдов добавляются все."""
        result = run_enforce(
            [],
            "rag_pipeline",
            "rag_pipeline/data=indexing",
            "trainer=fast",
        )
        assert "rag_pipeline/data=indexing" in result
        assert "trainer=fast" in result

    # ---------------------------------------------------------------------------
    # Логирование
    # ---------------------------------------------------------------------------

    def test_warning_logged_on_replace(self):
        """При замене pipeline_name логируется warning."""
        with patch("src.utils.cli.logger") as mock_logger:
            run_enforce(["pipeline_name=wrong"], "correct")
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0]
            assert "wrong" in args[1]
            assert "correct" in args[2]

    def test_info_logged_on_add_override(self):
        """При добавлении нового оверрайда логируется info."""
        with patch("src.utils.cli.logger") as mock_logger:
            run_enforce([], "rag_pipeline", "trainer=fast")
            assert any("trainer=fast" in str(call.args) for call in mock_logger.info.call_args_list)

    # ---------------------------------------------------------------------------
    # Без мутации оригинального sys.argv в других тестах
    # ---------------------------------------------------------------------------

    def test_does_not_mutate_argv_outside_patch(self):
        """После теста оригинальный sys.argv не изменён."""
        original = list(sys.argv)
        run_enforce([], "rag_pipeline")
        assert sys.argv == original
