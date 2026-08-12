from video2txt.align.normalize import normalize_text


def test_normalize_text_for_matching() -> None:
    assert normalize_text("ＡＩ，讓 開發 更簡單！") == "ai让开发更简单"


def test_normalize_can_preserve_traditional_chinese() -> None:
    assert normalize_text("軟體開發", simplify_chinese=False) == "軟體開發"

