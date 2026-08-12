from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from video2txt.config import OCRSettings
from video2txt.media.frames import extract_subtitle_frames
from video2txt.models import OCRLine, SourceType
from video2txt.ocr.hard_subtitles import extract_hard_subtitles
from video2txt.ocr.paddle import PaddleOCREngine


def _make_frame(path: Path, rectangle: tuple[int, int, int, int]) -> None:
    image = Image.new("RGB", (160, 50), "white")
    ImageDraw.Draw(image).rectangle(rectangle, fill="black")
    image.save(path)


def test_frame_extraction_uses_configured_crop_and_sampling(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    captured: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        captured.extend(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("video2txt.media.frames.subprocess.run", fake_run)
    settings = OCRSettings(sample_interval=1.25, crop_top=0.6, crop_bottom=0.95, scale=2)

    frames = extract_subtitle_frames(source, tmp_path / "frames", settings)

    assert frames == []
    video_filter = captured[captured.index("-vf") + 1]
    assert "crop=iw:trunc(ih*0.350000/2)*2" in video_filter
    assert "scale=iw*2:ih*2" in video_filter
    assert "fps=1/1.250000" in video_filter


def test_paddle_adapter_converts_result_lines(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    adapter = PaddleOCREngine(OCRSettings(min_confidence=0.6))
    adapter._engine = SimpleNamespace(
        predict=lambda _path: [
            {
                "rec_texts": ["保留", "忽略"],
                "rec_scores": [0.93, 0.41],
                "rec_polys": [
                    [[0, 0], [10, 0], [10, 4], [0, 4]],
                    [[0, 6], [10, 6], [10, 10], [0, 10]],
                ],
            }
        ]
    )

    lines = adapter.recognize(image)

    assert lines == [
        OCRLine(
            text="保留",
            confidence=0.93,
            bbox=[[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]],
        )
    ]


def test_hard_subtitle_extraction_skips_frames_and_merges_text(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    frames = [tmp_path / f"frame-{index:06d}.jpg" for index in range(1, 5)]
    _make_frame(frames[0], (20, 10, 80, 30))
    _make_frame(frames[1], (20, 10, 80, 30))
    _make_frame(frames[2], (25, 10, 85, 30))
    _make_frame(frames[3], (60, 10, 130, 30))
    monkeypatch.setattr(
        "video2txt.ocr.hard_subtitles.extract_subtitle_frames",
        lambda *_args, **_kwargs: frames,
    )
    texts = {
        frames[0].name: [OCRLine(text="你好", confidence=0.9)],
        frames[2].name: [OCRLine(text="你好！", confidence=0.85)],
        frames[3].name: [OCRLine(text="下一句", confidence=0.92)],
    }
    engine = SimpleNamespace(recognize=lambda path: texts[path.name])
    settings = OCRSettings(
        sample_interval=1,
        image_change_threshold=0.001,
        text_similarity_threshold=0.85,
    )

    progress = []
    result = extract_hard_subtitles(
        tmp_path / "input.mp4",
        tmp_path / "frames",
        settings,
        engine=engine,
        duration=4,
        progress_callback=progress.append,
    )

    assert result.frame_count == 4
    assert result.ocr_calls == 3
    assert result.observations[1].skipped_as_duplicate is True
    assert [cue.text for cue in result.cues] == ["你好", "下一句"]
    assert result.cues[0].start == 0
    assert result.cues[0].end == 3
    assert result.cues[0].source == SourceType.HARD_SUBTITLE
    assert progress[0].processed_frames == 0
    assert progress[0].total_frames == 4
    assert progress[-1].processed_frames == 4
    assert progress[-1].ocr_calls == 3
    assert progress[-1].skipped_frames == 1
    assert progress[-1].observation == result.observations[-1]


def test_hard_subtitle_extraction_filters_small_overlay_text(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    frame = tmp_path / "frame-000001.jpg"
    _make_frame(frame, (40, 20, 120, 42))
    monkeypatch.setattr(
        "video2txt.ocr.hard_subtitles.extract_subtitle_frames",
        lambda *_args, **_kwargs: [frame],
    )
    engine = SimpleNamespace(
        recognize=lambda _path: [
            OCRLine(
                text="MOMA",
                confidence=0.99,
                bbox=[[70, 8], [90, 8], [90, 12], [70, 12]],
            ),
            OCRLine(
                text="大家好，我是老陆",
                confidence=0.98,
                bbox=[[40, 20], [120, 20], [120, 42], [40, 42]],
            ),
        ]
    )

    result = extract_hard_subtitles(
        tmp_path / "input.mp4",
        tmp_path / "frames",
        OCRSettings(),
        engine=engine,
        duration=1.2,
    )

    assert result.ocr_calls == 1
    assert [line.text for line in result.observations[0].lines] == [
        "大家好，我是老陆"
    ]
    assert [cue.text for cue in result.cues] == ["大家好，我是老陆"]


def test_ocr_defaults_use_compact_subtitle_band() -> None:
    settings = OCRSettings()

    assert settings.sample_interval == 1.2
    assert settings.crop_top == 0.78
    assert settings.crop_bottom == 0.94
    assert settings.scale == 1
    assert settings.image_change_threshold == 0.12
