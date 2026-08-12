from pathlib import Path

import pytest

from video2txt.media.audio import AudioExtractionError, normalize_audio


def test_normalize_audio_builds_safe_argument_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input file.mp4"
    source.write_bytes(b"input")
    output = tmp_path / "audio.wav"
    captured: list[str] = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command: list[str], **_: object) -> Result:
        captured.extend(command)
        output.write_bytes(b"R" * 100)
        return Result()

    monkeypatch.setattr("video2txt.media.audio.subprocess.run", fake_run)

    result = normalize_audio(source, output, stream_index=2)

    assert result == output.resolve()
    assert captured[captured.index("-map") + 1] == "0:2"
    assert str(source.resolve()) in captured
    assert "-ar" in captured and "16000" in captured
    assert "-ac" in captured and "1" in captured


def test_normalize_audio_reports_ffmpeg_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"input")

    class Result:
        returncode = 1
        stderr = "missing audio stream"

    monkeypatch.setattr("video2txt.media.audio.subprocess.run", lambda *_args, **_kwargs: Result())

    with pytest.raises(AudioExtractionError, match="missing audio stream"):
        normalize_audio(source, tmp_path / "audio.wav")

