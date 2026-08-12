import json
from pathlib import Path

from video2txt.export.results import export_json, export_srt, export_text
from video2txt.models import AlignmentResult, FusionSegment, SourceType


def test_export_all_formats(tmp_path: Path) -> None:
    segments = [
        FusionSegment(
            start=1.0,
            end=2.5,
            text="第一句",
            source=SourceType.ASR,
            decision="asr_only",
        )
    ]
    alignment = AlignmentResult()

    text_path = export_text(segments, tmp_path / "result.txt")
    srt_path = export_srt(segments, tmp_path / "result.srt")
    json_path = export_json(alignment, segments, tmp_path / "result.json")

    assert text_path.read_text(encoding="utf-8") == "第一句\n"
    assert "00:00:01,000 --> 00:00:02,500" in srt_path.read_text(encoding="utf-8-sig")
    assert json.loads(json_path.read_text(encoding="utf-8"))["segments"][0]["text"] == "第一句"

