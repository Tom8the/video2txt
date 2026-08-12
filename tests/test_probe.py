from pathlib import Path

import pytest

from video2txt.media.probe import (
    classify_subtitle,
    parse_ffprobe,
    select_audio_stream,
    select_subtitle_stream,
)
from video2txt.models import SubtitleKind


@pytest.fixture
def ffprobe_payload() -> dict[str, object]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "tags": {"language": "eng"},
            },
            {
                "index": 2,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "tags": {"language": "zho"},
                "disposition": {"default": 1},
            },
            {
                "index": 3,
                "codec_type": "subtitle",
                "codec_name": "ass",
                "tags": {"language": "zho", "title": "简体中文"},
            },
            {
                "index": 4,
                "codec_type": "subtitle",
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "eng"},
            },
        ],
        "format": {
            "format_name": "matroska,webm",
            "duration": "12.345",
            "bit_rate": "1000000",
        },
    }


def test_parse_ffprobe_classifies_streams(ffprobe_payload: dict[str, object]) -> None:
    probe = parse_ffprobe(ffprobe_payload, path=Path("sample.mkv"), sha256="abc")

    assert probe.duration == 12.345
    assert len(probe.audio_streams) == 2
    assert probe.subtitle_streams[0].subtitle_kind == SubtitleKind.TEXT
    assert probe.subtitle_streams[1].subtitle_kind == SubtitleKind.IMAGE


def test_select_stream_prefers_language_then_default(ffprobe_payload: dict[str, object]) -> None:
    probe = parse_ffprobe(ffprobe_payload, path=Path("sample.mkv"), sha256="abc")

    assert select_audio_stream(probe).index == 2
    assert select_audio_stream(probe, language="eng").index == 1
    assert select_subtitle_stream(probe, language="zho").index == 3


def test_explicit_stream_index_wins(ffprobe_payload: dict[str, object]) -> None:
    probe = parse_ffprobe(ffprobe_payload, path=Path("sample.mkv"), sha256="abc")

    assert select_audio_stream(probe, stream_index=1).language == "eng"
    with pytest.raises(ValueError, match="索引"):
        select_audio_stream(probe, stream_index=99)


@pytest.mark.parametrize(
    ("codec", "expected"),
    [
        ("subrip", SubtitleKind.TEXT),
        ("mov_text", SubtitleKind.TEXT),
        ("dvd_subtitle", SubtitleKind.IMAGE),
        ("unknown_codec", SubtitleKind.UNKNOWN),
        (None, SubtitleKind.UNKNOWN),
    ],
)
def test_classify_subtitle(codec: str | None, expected: SubtitleKind) -> None:
    assert classify_subtitle(codec) == expected

