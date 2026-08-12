import json
from pathlib import Path

from video2txt.export.results import export_json, export_srt, export_text
from video2txt.models import AlignmentResult, FusionSegment, SourceType


def test_review_only_hard_subtitle_is_not_written_to_text_or_srt(
    tmp_path: Path,
) -> None:
    segments = [
        FusionSegment(
            start=1.0,
            end=2.0,
            text="仅供复核的硬字幕",
            source=SourceType.HARD_SUBTITLE,
            decision="hard_subtitle_unmatched_review",
            needs_review=True,
            include_in_transcript=False,
        )
    ]

    text_path = export_text(segments, tmp_path / "review.txt")
    srt_path = export_srt(segments, tmp_path / "review.srt")
    json_path = export_json(AlignmentResult(), segments, tmp_path / "review.json")

    assert text_path.read_text(encoding="utf-8") == "\n"
    assert "仅供复核的硬字幕" not in srt_path.read_text(encoding="utf-8-sig")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["segments"][0]["include_in_transcript"] is False
