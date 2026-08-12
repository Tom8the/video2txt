from video2txt.align.fusion import fuse_timeline
from video2txt.align.timeline import align_timeline
from video2txt.config import AlignmentSettings
from video2txt.models import (
    AlignmentGroup,
    AlignmentResult,
    FusionMode,
    SourceType,
    SubtitleCue,
    TranscriptSegment,
)


def test_verbatim_mode_retains_asr_when_subtitle_is_abbreviated() -> None:
    asr = [
        TranscriptSegment(
            id="a1", start=0, end=2, text="嗯我们今天开始开发", confidence=0.9
        )
    ]
    subtitles = [
        SubtitleCue(
            id="s1",
            start=0,
            end=2,
            text="今天开始开发",
            source=SourceType.EMBEDDED_TEXT,
            confidence=1,
        )
    ]
    settings = AlignmentSettings()
    alignment = align_timeline(asr, subtitles, settings)

    result = fuse_timeline(alignment, asr, subtitles, settings, mode=FusionMode.VERBATIM)

    assert result[0].text == "嗯我们今天开始开发"
    assert result[0].decision == "matched_asr_verbatim"


def test_subtitle_mode_prefers_subtitle() -> None:
    asr = [TranscriptSegment(id="a1", start=0, end=2, text="软件开发", confidence=0.9)]
    subtitles = [
        SubtitleCue(
            id="s1",
            start=0,
            end=2,
            text="软件开发。",
            source=SourceType.EMBEDDED_TEXT,
            confidence=1,
        )
    ]
    settings = AlignmentSettings()
    alignment = align_timeline(asr, subtitles, settings)

    result = fuse_timeline(alignment, asr, subtitles, settings, mode=FusionMode.SUBTITLE)

    assert result[0].text == "软件开发。"
    assert result[0].source == SourceType.MERGED


def test_verbatim_mode_uses_similar_subtitle_to_correct_names() -> None:
    asr = [
        TranscriptSegment(
            id="a1", start=0, end=2, text="TRwork Work Dadi Codex", confidence=0.9
        )
    ]
    subtitles = [
        SubtitleCue(
            id="s1",
            start=0,
            end=2,
            text="TRAE Work、WorkBuddy、Codex",
            source=SourceType.EMBEDDED_TEXT,
            confidence=1,
        )
    ]
    settings = AlignmentSettings()
    alignment = align_timeline(asr, subtitles, settings)

    result = fuse_timeline(alignment, asr, subtitles, settings, mode=FusionMode.VERBATIM)

    assert result[0].text == "TRAE Work、WorkBuddy、Codex"
    assert result[0].decision == "matched_subtitle_punctuation"


def test_unmatched_hard_subtitle_is_review_only_by_default() -> None:
    subtitles = [
        SubtitleCue(
            id="s1",
            start=1,
            end=2,
            text="画面中的课件文字",
            source=SourceType.HARD_SUBTITLE,
            confidence=0.98,
        )
    ]
    alignment = AlignmentResult(
        groups=[
            AlignmentGroup(
                subtitle_ids=["s1"],
                start=1,
                end=2,
                time_score=0,
                text_similarity=0,
                confidence_score=0.98,
                score=0.098,
                matched=False,
            )
        ]
    )

    result = fuse_timeline(
        alignment,
        [],
        subtitles,
        AlignmentSettings(),
        mode=FusionMode.VERBATIM,
    )

    assert result[0].text == "画面中的课件文字"
    assert result[0].decision == "hard_subtitle_unmatched_review"
    assert result[0].needs_review is True
    assert result[0].include_in_transcript is False


def test_matched_hard_subtitle_does_not_replace_asr_in_verbatim_mode() -> None:
    asr = [
        TranscriptSegment(
            id="a1", start=0, end=2, text="已经卖了几千份", confidence=0.9
        )
    ]
    subtitles = [
        SubtitleCue(
            id="s1",
            start=0,
            end=2,
            text="已经卖了几干份",
            source=SourceType.HARD_SUBTITLE,
            confidence=0.98,
        )
    ]
    settings = AlignmentSettings()
    alignment = align_timeline(asr, subtitles, settings)

    result = fuse_timeline(
        alignment, asr, subtitles, settings, mode=FusionMode.VERBATIM
    )

    assert result[0].text == "已经卖了几千份"
    assert result[0].decision == "matched_asr_hard_subtitle_reference"
