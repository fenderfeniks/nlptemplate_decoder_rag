from src.core.models.parsers import ResponseCleaner


def test_response_cleaner_removes_special_tokens():
    cleaner = ResponseCleaner()
    dirty_text = (
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "Привет! Это ответ.<|eot_id|>"
    )

    cleaned = cleaner.clean(dirty_text)

    assert cleaned == "Привет! Это ответ."


def test_response_cleaner_strips_extra_whitespace():
    cleaner = ResponseCleaner(trim_incomplete_sentence=False)
    dirty_text = "   \n\nПривет!\n\n\n  "

    cleaned = cleaner.clean(dirty_text)

    assert cleaned == "Привет!"
