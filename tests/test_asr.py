from pathlib import Path
from types import SimpleNamespace

import pytest

from video2txt.asr.faster_whisper import ASRError, FasterWhisperEngine
from video2txt.config import ASRSettings


class FakeModel:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def transcribe(self, _audio: str, **options: object):  # type: ignore[no-untyped-def]
        self.options = options
        word = SimpleNamespace(start=0.12, end=0.48, word=" 你好", probability=0.96)
        segment = SimpleNamespace(
            start=0.1,
            end=0.6,
            text=" 你好",
            avg_logprob=-0.1,
            words=[word],
        )
        return iter([segment]), SimpleNamespace(language="zh")


def test_transcribe_persists_word_timestamps_and_options(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not-real-audio")
    model = FakeModel()
    settings = ASRSettings(model_path=tmp_path / "model")

    transcript = FasterWhisperEngine(settings, model=model).transcribe(audio)

    assert transcript.segments[0].text == "你好"
    assert transcript.segments[0].words[0].start == 0.12
    assert transcript.segments[0].words[0].probability == 0.96
    assert transcript.options["beam_size"] == 5
    assert model.options["word_timestamps"] is True
    assert len(transcript.audio_sha256) == 64


def test_empty_transcription_is_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not-real-audio")

    class EmptyModel:
        def transcribe(self, *_args: object, **_kwargs: object):
            return iter([]), SimpleNamespace(language="zh")

    with pytest.raises(ASRError, match="未识别"):
        FasterWhisperEngine(ASRSettings(), model=EmptyModel()).transcribe(audio)

