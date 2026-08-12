from pathlib import Path

from video2txt.cleaning import clean_fusion_segments
from video2txt.export.results import export_text
from video2txt.models import FusionSegment, SourceType


def _segment(start: float, end: float, text: str) -> FusionSegment:
    return FusionSegment(
        start=start,
        end=end,
        text=text,
        source=SourceType.ASR,
        decision="asr_only",
    )


def test_cleaning_removes_fillers_duplicates_and_adds_punctuation(
    tmp_path: Path,
) -> None:
    source = [
        _segment(0, 1, "嗯，大家好 我是老陆"),
        _segment(1.1, 2, "大家好 我是老陆"),
        _segment(2.1, 3, "呃"),
        _segment(3.1, 4, "然后 然后 我们开始"),
        _segment(7, 8, "这样可以吗"),
    ]

    cleaned = clean_fusion_segments(
        source, paragraph_max_chars=30, paragraph_gap_seconds=2
    )
    included = [item for item in cleaned.segments if item.include_in_transcript]

    assert [item.text for item in included] == [
        "大家好我是老陆。",
        "然后我们开始。",
        "这样可以吗？",
    ]
    assert cleaned.removed_duplicates == 1
    assert cleaned.removed_fillers == 1
    assert cleaned.paragraph_count == 2
    assert cleaned.segments[0].original_text == "嗯，大家好 我是老陆"
    assert cleaned.segments[1].decision == "clean_removed_duplicate"
    assert cleaned.segments[2].decision == "clean_removed_filler"

    output = export_text(cleaned.segments, tmp_path / "clean.txt")
    assert output.read_text(encoding="utf-8") == (
        "大家好我是老陆。然后我们开始。\n\n这样可以吗？\n"
    )
