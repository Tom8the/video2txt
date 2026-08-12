from pathlib import Path

import pytest

from video2txt.media.subtitles import SubtitleExtractionError, extract_text_subtitle
from video2txt.models import SourceType
from video2txt.subtitles.parser import parse_subtitle_file


def test_parse_srt_preserves_raw_and_clean_text(tmp_path: Path) -> None:
    subtitle = tmp_path / "sample.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:03,250\n<b>你好</b>\n世界\n\n",
        encoding="utf-8",
    )

    cues = parse_subtitle_file(subtitle, language="zh")

    assert len(cues) == 1
    assert cues[0].start == 1.0
    assert cues[0].end == 3.25
    assert cues[0].text == "你好\n世界"
    assert cues[0].raw_text == "{\\b1}你好{\\b0}\\N世界"
    assert cues[0].source == SourceType.EXTERNAL_SUBTITLE


def test_parse_gb18030_srt(tmp_path: Path) -> None:
    subtitle = tmp_path / "gb.srt"
    subtitle.write_bytes("1\n00:00:00,000 --> 00:00:01,000\n中文测试\n".encode("gb18030"))

    assert parse_subtitle_file(subtitle)[0].text == "中文测试"


def test_extract_subtitle_uses_global_stream_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.mkv"
    source.write_bytes(b"input")
    output = tmp_path / "subtitle.srt"
    captured: list[str] = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command: list[str], **_: object) -> Result:
        captured.extend(command)
        output.write_text("subtitle", encoding="utf-8")
        return Result()

    monkeypatch.setattr("video2txt.media.subtitles.subprocess.run", fake_run)

    extract_text_subtitle(source, output, stream_index=3)

    assert captured[captured.index("-map") + 1] == "0:3"
    assert captured[captured.index("-c:s") + 1] == "srt"


def test_extract_subtitle_reports_ffmpeg_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.mkv"
    source.write_bytes(b"input")

    class Result:
        returncode = 1
        stderr = "subtitle encoder failure"

    monkeypatch.setattr(
        "video2txt.media.subtitles.subprocess.run", lambda *_args, **_kwargs: Result()
    )

    with pytest.raises(SubtitleExtractionError, match="encoder failure"):
        extract_text_subtitle(source, tmp_path / "subtitle.srt")
