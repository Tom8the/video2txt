import pytest
from pydantic import ValidationError

from video2txt.models import SourceType, TranscriptSegment, TranscriptWord


def test_transcript_segment_preserves_word_timestamps() -> None:
    segment = TranscriptSegment(
        id="asr-0001",
        start=0.5,
        end=1.2,
        text="你好",
        confidence=0.93,
        words=[TranscriptWord(start=0.5, end=1.2, text="你好", probability=0.97)],
    )

    assert segment.words[0].probability == 0.97
    assert SourceType.ASR.value == "asr"


def test_invalid_word_time_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TranscriptWord(start=2.0, end=1.0, text="错误")

