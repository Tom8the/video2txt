from video2txt.align.timeline import (
    align_timeline,
    estimate_subtitle_offset,
    interval_overlap_score,
    text_similarity,
)
from video2txt.config import AlignmentSettings
from video2txt.models import SourceType, SubtitleCue, TranscriptSegment


def asr(item_id: str, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        id=item_id, start=start, end=end, text=text, confidence=0.9
    )


def subtitle(item_id: str, start: float, end: float, text: str) -> SubtitleCue:
    return SubtitleCue(
        id=item_id,
        start=start,
        end=end,
        text=text,
        source=SourceType.EMBEDDED_TEXT,
        confidence=1.0,
    )


def test_similarity_normalizes_traditional_and_punctuation() -> None:
    assert text_similarity("軟件開發！", "软件开发") == 1.0


def test_interval_overlap_uses_shorter_duration() -> None:
    assert interval_overlap_score(0, 4, 1, 3) == 1.0
    assert interval_overlap_score(0, 1, 2, 3) == 0.0


def test_estimate_subtitle_offset() -> None:
    result = estimate_subtitle_offset(
        [asr("a1", 0, 2, "第一句"), asr("a2", 3, 5, "第二句")],
        [subtitle("s1", 1, 3, "第一句"), subtitle("s2", 4, 6, "第二句")],
    )
    assert result == 1.0


def test_align_supports_one_asr_to_multiple_subtitles() -> None:
    result = align_timeline(
        [asr("a1", 0, 4, "第一句话第二句话")],
        [subtitle("s1", 0, 2, "第一句话"), subtitle("s2", 2, 4, "第二句话")],
        AlignmentSettings(),
    )

    assert len(result.groups) == 1
    assert result.groups[0].asr_ids == ["a1"]
    assert result.groups[0].subtitle_ids == ["s1", "s2"]
    assert result.groups[0].matched is True


def test_unmatched_segments_are_preserved() -> None:
    result = align_timeline(
        [asr("a1", 0, 1, "只有语音")],
        [subtitle("s1", 5, 6, "只有字幕")],
        AlignmentSettings(),
    )

    assert len(result.groups) == 2
    assert all(group.matched is False for group in result.groups)


def test_adjacent_pairs_do_not_collapse_into_one_large_group() -> None:
    asr_items = [asr(f"a{i}", i * 2, i * 2 + 1.8, f"第{i}句") for i in range(5)]
    subtitle_items = [
        subtitle(f"s{i}", i * 2, i * 2 + 1.8, f"第{i}句") for i in range(5)
    ]

    result = align_timeline(asr_items, subtitle_items, AlignmentSettings())

    assert len(result.groups) == 5
    assert all(len(group.asr_ids) == 1 for group in result.groups)
    assert all(len(group.subtitle_ids) == 1 for group in result.groups)
