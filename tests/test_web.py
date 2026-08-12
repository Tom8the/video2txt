import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Event
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from video2txt.config import Settings
from video2txt.models import FusionMode, TaskManifest, TaskStatus
from video2txt.web.app import (
    MAX_UPLOAD_BYTES,
    _task_job_path,
    _write_batch_manifest,
    _write_json_atomic,
    create_app,
)


def spin_cpu(seconds: float) -> int:
    deadline = time.perf_counter() + seconds
    iterations = 0
    while time.perf_counter() < deadline:
        iterations += 1
    return iterations


def slow_task_runner(
    settings_payload: dict[str, object], job: dict[str, object]
) -> str | None:
    spin_cpu(1.5)
    paths = dict(settings_payload["paths"])  # type: ignore[arg-type]
    marker = Path(str(paths["work_dir"])) / "worker-finished.txt"
    marker.write_text(str(job["task_id"]), encoding="utf-8")
    return None


def configured_settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.paths.work_dir = tmp_path / "work"
    settings.paths.output_dir = tmp_path / "output"
    settings.asr.model_path = tmp_path / "model"
    settings.asr.model_path.mkdir()
    return settings


def write_task_manifest(
    settings: Settings,
    task_id: str,
    source: Path,
    status: TaskStatus,
    *,
    batch_id: str | None = None,
) -> TaskManifest:
    work_dir = settings.paths.work_dir / task_id
    output_dir = settings.paths.output_dir / task_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = TaskManifest(
        task_id=task_id,
        status=status,
        input_path=source,
        original_filename=source.name,
        batch_id=batch_id,
        work_dir=work_dir,
        output_dir=output_dir,
        mode=FusionMode.VERBATIM,
        created_at="2026-07-21T10:00:00+08:00",
        updated_at="2026-07-21T10:01:00+08:00",
    )
    for target in (work_dir / "task.json", output_dir / "task.json"):
        target.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def test_web_home_and_health(tmp_path: Path) -> None:
    with TestClient(create_app(configured_settings(tmp_path))) as client:
        page = client.get("/")
        script = client.get("/app.js?v=20260723-batch-reset")
        health = client.get("/api/health")

    assert page.status_code == 200
    assert "把视频里的声音" in page.text
    assert "20260723-batch-reset" in page.text
    assert script.status_code == 200
    assert "根据历史速度估算" not in script.text
    assert 'class="history-download"' in script.text
    assert "export?types=text&types=subtitle" in script.text
    assert "closeHistoryDownloadMenus" in script.text
    assert "viewTask(button.dataset.taskId, true)" in script.text
    assert '"← 返回任务队列" : "← 返回最近任务"' in script.text
    assert "function resetBatchProgress(expectedTotal)" in script.text
    assert "resetBatchProgress(mediaFiles.length)" in script.text
    assert '$("#batch-completed").textContent = "0"' in script.text
    assert '$("#batch-progress-bar").style.width = "0%"' in script.text
    assert health.json()["model_configured"] is True
    assert health.json()["model_name"] == "model"


def test_health_uses_readable_hugging_face_model_name(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    model_path = (
        tmp_path
        / "models--Systran--faster-whisper-small"
        / "snapshots"
        / "536b0662742c02347bc0e980a01041f333bce120"
    )
    model_path.mkdir(parents=True)
    settings.asr.model_path = model_path

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")

    assert health.json()["model_name"] == "faster-whisper-small"


def test_worker_process_keeps_health_responsive(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    marker = settings.paths.work_dir / "worker-finished.txt"
    app = create_app(settings, task_runner=slow_task_runner)
    with TestClient(app) as client:
        executor = app.state.executor
        assert isinstance(executor, ProcessPoolExecutor)
        submitted = client.post(
            "/api/tasks",
            files={"media": ("lesson.wav", b"audio", "audio/wav")},
            data={"mode": "verbatim"},
        )
        started = time.perf_counter()
        health = client.get("/api/health")
        elapsed = time.perf_counter() - started
        deadline = time.perf_counter() + 5
        while not marker.is_file() and time.perf_counter() < deadline:
            time.sleep(0.05)

    assert submitted.status_code == 202
    assert health.status_code == 200
    assert marker.is_file()
    assert elapsed < 1.0


def test_upload_rejects_unsupported_media(tmp_path: Path) -> None:
    with TestClient(create_app(configured_settings(tmp_path))) as client:
        response = client.post(
            "/api/tasks",
            files={"media": ("notes.txt", b"not media", "text/plain")},
            data={"mode": "verbatim"},
        )

    assert response.status_code == 400
    assert "不支持" in response.json()["detail"]


def test_upload_limit_is_five_gibibytes() -> None:
    assert MAX_UPLOAD_BYTES == 5 * 1024 * 1024 * 1024


def test_multipart_temp_files_use_project_work_drive(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    app = create_app(settings)

    expected = settings.paths.work_dir.resolve() / "multipart"
    assert app.state.multipart_temp_dir == expected
    assert Path(str(tempfile.tempdir)) == expected
    assert expected.is_dir()


def test_body_parse_error_has_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_form(_request: Request, **_kwargs: object) -> object:
        raise OSError("temporary upload storage is full")

    monkeypatch.setattr(Request, "_get_form", broken_form)
    with TestClient(create_app(configured_settings(tmp_path))) as client:
        response = client.post(
            "/api/batches",
            files={"media": ("lesson.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "上传请求解析失败，请检查项目磁盘空间后重新选择文件上传"


def test_batch_upload_creates_tasks_and_pairs_subtitles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str | None, bool]] = []
    completed = Event()

    def fake_run(_self: object, input_path: Path, **kwargs: object) -> None:
        subtitle = kwargs.get("external_subtitle")
        calls.append(
            (
                input_path.suffix,
                Path(str(subtitle)).suffix if subtitle else None,
                bool(kwargs.get("hard_subtitles")),
            )
        )
        if len(calls) == 2:
            completed.set()

    monkeypatch.setattr("video2txt.web.app.TranscriptionPipeline.run", fake_run)
    with TestClient(
        create_app(
            configured_settings(tmp_path),
            task_executor=ThreadPoolExecutor(max_workers=1),
        )
    ) as client:
        response = client.post(
            "/api/batches",
            files=[
                ("media", ("lesson-1.wav", b"audio-one", "audio/wav")),
                ("media", ("lesson-2.mp3", b"audio-two", "audio/mpeg")),
                ("subtitles", ("lesson-1.srt", b"subtitle", "text/plain")),
                ("subtitles", ("", b"", "application/octet-stream")),
            ],
            data={"mode": "verbatim", "hard_subtitles": "true"},
        )
        assert response.status_code == 202, response.text
        assert completed.wait(2)
        batch = client.get(f"/api/batches/{response.json()['batch_id']}")

    assert [task["original_filename"] for task in response.json()["tasks"]] == [
        "lesson-1.wav",
        "lesson-2.mp3",
    ]
    assert batch.status_code == 200
    assert len(batch.json()["tasks"]) == 2
    assert calls == [(".wav", ".srt", True), (".mp3", None, True)]


def test_batch_upload_rejects_unmatched_subtitle(tmp_path: Path) -> None:
    with TestClient(create_app(configured_settings(tmp_path))) as client:
        response = client.post(
            "/api/batches",
            files=[
                ("media", ("lesson.wav", b"audio", "audio/wav")),
                ("subtitles", ("other.srt", b"subtitle", "text/plain")),
            ],
            data={"mode": "verbatim"},
        )

    assert response.status_code == 400
    assert "字幕没有同名媒体" in response.json()["detail"]


def test_batch_export_packages_completed_outputs(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    task_id = "task-one"
    batch_id = "batch-one"
    output_dir = settings.paths.output_dir / task_id
    work_dir = settings.paths.work_dir / task_id
    output_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    source = tmp_path / "访谈.wav"
    source.write_bytes(b"audio")
    (output_dir / "transcript.txt").write_text("测试文本", encoding="utf-8")
    (output_dir / "subtitles.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n测试\n", encoding="utf-8"
    )
    (output_dir / "task.json").write_text("{}", encoding="utf-8")
    manifest = TaskManifest(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        input_path=source,
        original_filename=source.name,
        batch_id=batch_id,
        work_dir=work_dir,
        output_dir=output_dir,
        mode=FusionMode.VERBATIM,
        created_at="2026-07-20T12:00:00+08:00",
        updated_at="2026-07-20T12:01:00+08:00",
    )
    (output_dir / "task.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    _write_batch_manifest(
        settings,
        {
            "batch_id": batch_id,
            "mode": "verbatim",
            "created_at": "2026-07-20T12:00:00+08:00",
            "task_ids": [task_id],
        },
    )

    with TestClient(create_app(settings)) as client:
        response = client.get(
            f"/api/batches/{batch_id}/export.zip?types=text&types=subtitle"
        )
        text_response = client.get(f"/api/batches/{batch_id}/export.zip?types=text")
        subtitle_response = client.get(
            f"/api/batches/{batch_id}/export.zip?types=subtitle"
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            f"{source.stem}/{source.stem}.srt",
            f"{source.stem}/{source.stem}.txt",
        }
    with ZipFile(BytesIO(text_response.content)) as archive:
        assert archive.namelist() == [f"{source.stem}/{source.stem}.txt"]
    with ZipFile(BytesIO(subtitle_response.content)) as archive:
        assert archive.namelist() == [f"{source.stem}/{source.stem}.srt"]


def test_task_export_uses_video_name_and_excludes_json(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    task_id = "export-task"
    source = tmp_path / "lesson.mp4"
    source.write_bytes(b"video")
    manifest = write_task_manifest(settings, task_id, source, TaskStatus.COMPLETED)
    (manifest.output_dir / "transcript.txt").write_text("text", encoding="utf-8")
    (manifest.output_dir / "subtitles.srt").write_text("subtitle", encoding="utf-8")
    (manifest.output_dir / "fusion.json").write_text("{}", encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        all_response = client.get(
            f"/api/tasks/{task_id}/export?types=text&types=subtitle"
        )
        text_response = client.get(f"/api/tasks/{task_id}/export")
        subtitle_response = client.get(
            f"/api/tasks/{task_id}/export?types=subtitle"
        )

    assert all_response.status_code == 200
    assert "lesson.zip" in all_response.headers["content-disposition"]
    with ZipFile(BytesIO(all_response.content)) as archive:
        assert set(archive.namelist()) == {"lesson.txt", "lesson.srt"}
    assert text_response.content == b"text"
    assert "lesson.txt" in text_response.headers["content-disposition"]
    assert subtitle_response.content == b"subtitle"
    assert "lesson.srt" in subtitle_response.headers["content-disposition"]


def test_download_is_restricted_to_known_outputs(tmp_path: Path) -> None:
    with TestClient(create_app(configured_settings(tmp_path))) as client:
        response = client.get("/api/tasks/example/files/unknown.bin")
        json_response = client.get("/api/tasks/example/files/fusion.json")

    assert response.status_code == 404
    assert json_response.status_code == 404


def test_delete_terminal_task_removes_managed_files(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    task_id = "delete-task"
    batch_id = "delete-batch"
    source = settings.paths.work_dir / "uploads" / batch_id / task_id / "source.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    manifest = write_task_manifest(
        settings, task_id, source, TaskStatus.COMPLETED, batch_id=batch_id
    )
    (manifest.work_dir / "audio.wav").write_bytes(b"derived")
    (manifest.output_dir / "transcript.txt").write_text("text", encoding="utf-8")
    _write_batch_manifest(
        settings,
        {
            "batch_id": batch_id,
            "mode": "verbatim",
            "created_at": "2026-07-21T10:00:00+08:00",
            "task_ids": [task_id],
        },
    )

    with TestClient(create_app(settings)) as client:
        response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 200
    assert response.json()["freed_bytes"] > 0
    assert not manifest.work_dir.exists()
    assert not manifest.output_dir.exists()
    assert not source.parent.exists()
    assert not (settings.paths.work_dir / "batches" / f"{batch_id}.json").exists()


def test_delete_rejects_running_task(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    task_id = "running-task"
    source = settings.paths.work_dir / "uploads" / task_id / task_id / "source.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    manifest = write_task_manifest(settings, task_id, source, TaskStatus.TRANSCRIBING)

    with TestClient(create_app(settings)) as client:
        response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 409
    assert manifest.work_dir.is_dir()
    assert manifest.output_dir.is_dir()
    assert source.is_file()


def test_delete_all_tasks_clears_outputs_and_preserves_cache(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    batch_id = "clear-batch"
    manifests: list[TaskManifest] = []
    for task_id in ("clear-one", "clear-two"):
        source = settings.paths.work_dir / "uploads" / batch_id / task_id / "source.wav"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        manifest = write_task_manifest(
            settings,
            task_id,
            source,
            TaskStatus.COMPLETED,
            batch_id=batch_id,
        )
        (manifest.output_dir / "transcript.txt").write_text("text", encoding="utf-8")
        manifests.append(manifest)
    _write_batch_manifest(
        settings,
        {
            "batch_id": batch_id,
            "mode": "verbatim",
            "created_at": "2026-07-22T10:00:00+08:00",
            "task_ids": [manifest.task_id for manifest in manifests],
        },
    )
    cache = settings.paths.work_dir / "cache" / "asr" / "entry.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{}", encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        response = client.delete("/api/tasks")
        history = client.get("/api/tasks")

    assert response.status_code == 200
    assert response.json()["cleared_tasks"] == 2
    assert history.json()["tasks"] == []
    assert not any(settings.paths.output_dir.iterdir())
    assert not any((settings.paths.work_dir / "uploads").iterdir())
    assert not any((settings.paths.work_dir / "batches").iterdir())
    assert cache.is_file()


def test_delete_all_tasks_rejects_running_task(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    task_id = "running-clear"
    source = settings.paths.work_dir / "uploads" / task_id / task_id / "source.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    manifest = write_task_manifest(settings, task_id, source, TaskStatus.TRANSCRIBING)

    with TestClient(create_app(settings)) as client:
        response = client.delete("/api/tasks")

    assert response.status_code == 409
    assert manifest.output_dir.is_dir()
    assert source.is_file()


def test_failed_task_retry_copies_source_and_submits_new_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    task_id = "failed-retry"
    source = settings.paths.work_dir / "uploads" / task_id / task_id / "source.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"retry-audio")
    write_task_manifest(settings, task_id, source, TaskStatus.FAILED)
    completed = Event()
    captured: dict[str, object] = {}

    def fake_run(_self: object, input_path: Path, **kwargs: object) -> None:
        captured["input_path"] = input_path
        captured["task_id"] = kwargs["task_id"]
        completed.set()

    monkeypatch.setattr("video2txt.web.app.TranscriptionPipeline.run", fake_run)
    with TestClient(
        create_app(settings, task_executor=ThreadPoolExecutor(max_workers=1))
    ) as client:
        response = client.post(f"/api/tasks/{task_id}/retry")
        assert completed.wait(2)

    assert response.status_code == 202
    new_task_id = response.json()["task_ids"][0]
    copied_source = Path(str(captured["input_path"]))
    assert captured["task_id"] == new_task_id
    assert copied_source != source
    assert copied_source.read_bytes() == b"retry-audio"


def test_lifespan_recovers_persisted_queue_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    task_id = "recover-task"
    source = settings.paths.work_dir / "uploads" / task_id / task_id / "source.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"recover")
    _write_json_atomic(
        _task_job_path(settings, task_id),
        {
            "task_id": task_id,
            "batch_id": None,
            "status": "queued",
            "mode": "verbatim",
            "hard_subtitles": False,
            "original_filename": "source.wav",
            "media_size": source.stat().st_size,
            "warnings": [],
            "error": None,
            "media_path": str(source),
            "subtitle_path": None,
        },
    )
    completed = Event()

    def fake_run(_self: object, _input_path: Path, **kwargs: object) -> None:
        assert kwargs["task_id"] == task_id
        completed.set()

    monkeypatch.setattr("video2txt.web.app.TranscriptionPipeline.run", fake_run)
    with TestClient(
        create_app(settings, task_executor=ThreadPoolExecutor(max_workers=1))
    ):
        assert completed.wait(2)

    assert not _task_job_path(settings, task_id).exists()


def test_storage_summary_and_cleanup_remove_only_derived_files(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    task_id = "storage-task"
    source = settings.paths.work_dir / "uploads" / task_id / task_id / "source.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    manifest = write_task_manifest(settings, task_id, source, TaskStatus.COMPLETED)
    (manifest.work_dir / "audio.wav").write_bytes(b"audio-cache")
    frame = manifest.work_dir / "ocr-frames" / "frame-000001.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"frame")
    cache = settings.paths.work_dir / "cache" / "asr" / "entry.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"cached": True}), encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        summary = client.get("/api/storage")
        temporary = client.post(
            "/api/storage/cleanup", data={"scope": "temporary"}
        )
        cache_cleanup = client.post("/api/storage/cleanup", data={"scope": "cache"})

    assert summary.status_code == 200
    assert summary.json()["uploads_bytes"] == len(b"source")
    assert temporary.json()["freed_bytes"] == len(b"audio-cache") + len(b"frame")
    assert source.is_file()
    assert not (manifest.work_dir / "audio.wav").exists()
    assert not (manifest.work_dir / "ocr-frames").exists()
    assert cache_cleanup.json()["freed_bytes"] > 0
    assert not cache.exists()
