from pathlib import Path
from types import SimpleNamespace

from video2txt.config import Settings
from video2txt.models import (
    MediaProbe,
    MediaStream,
    OCRObservation,
    SourceType,
    StreamKind,
    SubtitleCue,
    TaskStatus,
    Transcript,
    TranscriptSegment,
)
from video2txt.ocr.hard_subtitles import HardSubtitleProgress, HardSubtitleResult
from video2txt.pipeline import TranscriptionPipeline


def test_pipeline_runs_all_stages_with_external_subtitle(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "input.mp4"
    source.write_bytes(b"media")
    external_subtitle = tmp_path / "input.srt"
    external_subtitle.write_text("subtitle", encoding="utf-8")
    settings = Settings()
    settings.paths.work_dir = tmp_path / "work"
    settings.paths.output_dir = tmp_path / "output"

    probe = MediaProbe(
        path=source,
        sha256="abc",
        duration=2,
        streams=[MediaStream(index=1, kind=StreamKind.AUDIO, codec_name="aac")],
    )
    monkeypatch.setattr("video2txt.pipeline.probe_media", lambda *_args, **_kwargs: probe)

    def fake_normalize(_source: Path, output: Path, **_kwargs: object) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"audio")
        return output

    monkeypatch.setattr("video2txt.pipeline.normalize_audio", fake_normalize)
    monkeypatch.setattr(
        "video2txt.pipeline.parse_subtitle_file",
        lambda *_args, **_kwargs: [
            SubtitleCue(
                id="sub-0001",
                start=0,
                end=2,
                text="准确字幕",
                source=SourceType.EXTERNAL_SUBTITLE,
                confidence=1,
            )
        ],
    )
    transcript = Transcript(
        engine="fake",
        model="fake",
        language="zh",
        audio_sha256="hash",
        segments=[
            TranscriptSegment(
                id="asr-0001", start=0, end=2, text="准确字幕", confidence=0.9
            )
        ],
    )
    monkeypatch.setattr(
        "video2txt.pipeline.FasterWhisperEngine",
        lambda _settings: SimpleNamespace(transcribe=lambda _audio: transcript),
    )

    result = TranscriptionPipeline(settings).run(
        source,
        external_subtitle=external_subtitle,
        task_id="test-task",
    )

    assert result.status == TaskStatus.COMPLETED
    assert result.selected_audio_stream == 1
    assert (result.output_dir / "transcript.txt").read_text(encoding="utf-8") == "准确字幕\n"
    assert (result.output_dir / "asr.json").is_file()
    assert (result.output_dir / "subtitle_raw.json").is_file()
    assert (result.output_dir / "fusion.json").is_file()
    assert (result.output_dir / "subtitles.srt").is_file()
    assert (result.output_dir / "task.json").is_file()


def test_pipeline_marks_manifest_failed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"broken")
    settings = Settings()
    settings.paths.work_dir = tmp_path / "work"
    settings.paths.output_dir = tmp_path / "output"
    monkeypatch.setattr(
        "video2txt.pipeline.probe_media",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )

    try:
        TranscriptionPipeline(settings).run(source, task_id="failed-task")
    except RuntimeError:
        pass

    manifest_path = tmp_path / "output" / "failed-task" / "task.json"
    assert manifest_path.is_file()
    manifest = manifest_path.read_text(encoding="utf-8")
    assert '"status": "failed"' in manifest
    assert "probe failed" in manifest


def test_pipeline_uses_hard_subtitle_ocr_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "hard-subtitles.mp4"
    source.write_bytes(b"media")
    settings = Settings()
    settings.paths.work_dir = tmp_path / "work"
    settings.paths.output_dir = tmp_path / "output"
    probe = MediaProbe(
        path=source,
        sha256="abc",
        duration=2,
        streams=[
            MediaStream(index=0, kind=StreamKind.VIDEO, codec_name="h264"),
            MediaStream(index=1, kind=StreamKind.AUDIO, codec_name="aac"),
        ],
    )
    monkeypatch.setattr("video2txt.pipeline.probe_media", lambda *_args, **_kwargs: probe)

    def fake_normalize(_source: Path, output: Path, **_kwargs: object) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"audio")
        return output

    monkeypatch.setattr("video2txt.pipeline.normalize_audio", fake_normalize)
    transcript = Transcript(
        engine="fake",
        model="fake",
        language="zh",
        audio_sha256="hash",
        segments=[
            TranscriptSegment(id="asr-0001", start=0, end=2, text="画面字幕", confidence=0.9)
        ],
    )
    monkeypatch.setattr(
        TranscriptionPipeline,
        "_transcribe_with_cache",
        lambda *_args, **_kwargs: transcript,
    )
    cue = SubtitleCue(
        id="ocr-0001",
        start=0,
        end=2,
        text="画面字幕",
        source=SourceType.HARD_SUBTITLE,
        confidence=0.88,
    )
    observation = OCRObservation(
        frame="frame-000003.jpg",
        timestamp=1.6,
        image_difference=0.2,
    )

    def fake_extract(*_args: object, **kwargs: object) -> HardSubtitleResult:
        callback = kwargs["progress_callback"]
        callback(
            HardSubtitleProgress(
                total_frames=3,
                processed_frames=0,
                ocr_calls=0,
                skipped_frames=0,
            )
        )
        callback(
            HardSubtitleProgress(
                total_frames=3,
                processed_frames=3,
                ocr_calls=2,
                skipped_frames=1,
                observation=observation,
            )
        )
        partial_path = settings.paths.output_dir / "ocr-task" / "ocr_observations.json"
        assert partial_path.is_file()
        assert "frame-000003.jpg" in partial_path.read_text(encoding="utf-8")
        return HardSubtitleResult(
            cues=[cue], observations=[observation], frame_count=3, ocr_calls=2
        )

    monkeypatch.setattr("video2txt.pipeline.extract_hard_subtitles", fake_extract)

    result = TranscriptionPipeline(settings).run(
        source,
        task_id="ocr-task",
        hard_subtitles=True,
    )

    assert result.status == TaskStatus.COMPLETED
    assert result.hard_subtitles is True
    assert result.progress is not None
    assert result.progress.current == 3
    assert result.progress.total == 3
    assert result.progress.ocr_calls == 2
    assert result.progress.skipped == 1
    assert (result.output_dir / "ocr_observations.json").is_file()
    assert "frame-000003.jpg" in (
        result.output_dir / "ocr_observations.json"
    ).read_text(encoding="utf-8")
    subtitle_payload = (result.output_dir / "subtitle_raw.json").read_text(encoding="utf-8")
    assert '"source": "hard_subtitle"' in subtitle_payload
    assert any("执行 2 次识别" in warning for warning in result.warnings)
