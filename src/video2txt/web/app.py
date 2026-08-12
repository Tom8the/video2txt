from __future__ import annotations

import importlib.util
import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from video2txt import __version__
from video2txt.config import Settings, load_settings
from video2txt.models import FusionMode, TaskManifest, TaskStatus
from video2txt.pipeline import TranscriptionPipeline

MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}
SUBTITLE_EXTENSIONS = {".ass", ".srt", ".ssa", ".vtt"}
DOWNLOAD_FILES = {"subtitles.srt", "transcript.txt"}
EXPORT_FILES_BY_TYPE = {
    "text": "transcript.txt",
    "subtitle": "subtitles.srt",
}
EXPORT_FILES_BY_KIND = {
    "all": ("transcript.txt", "subtitles.srt"),
    "subtitle": ("subtitles.srt",),
    "text": ("transcript.txt",),
}
ExportType = Literal["subtitle", "text"]
LegacyExportKind = Literal["all", "subtitle", "text"]
MAX_UPLOAD_GB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_GB * 1024 * 1024 * 1024
MAX_BATCH_FILES = 30
CHUNK_SIZE = 1024 * 1024
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}


def _model_display_name(model_path: Path | None) -> str | None:
    if model_path is None:
        return None
    for part in reversed(model_path.parts):
        prefix = "models--Systran--"
        if part.startswith(prefix):
            return part.removeprefix(prefix)
        if part.startswith("faster-whisper-"):
            return part
    return model_path.name


def _validate_id(value: str, label: str) -> str:
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise HTTPException(status_code=404, detail=f"{label}ä¸å­˜åœ¨")
    return value


def _batch_manifest_path(settings: Settings, batch_id: str) -> Path:
    return settings.paths.work_dir.resolve() / "batches" / f"{batch_id}.json"


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_batch_manifest(settings: Settings, payload: dict[str, Any]) -> None:
    path = _batch_manifest_path(settings, payload["batch_id"])
    _write_json_atomic(path, payload)


def _task_job_path(settings: Settings, task_id: str) -> Path:
    return settings.paths.work_dir.resolve() / "queue" / f"{task_id}.json"


def _directory_size(path: Path) -> int:
    if not path.exists() or path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _remove_child_path(root: Path, target: Path) -> int:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise RuntimeError(f"refusing to remove path outside managed root: {target}")
    size = _directory_size(resolved_target)
    if resolved_target.is_dir() and not resolved_target.is_symlink():
        shutil.rmtree(resolved_target)
    elif resolved_target.exists():
        resolved_target.unlink()
    return size


def _read_batch_manifest(settings: Settings, batch_id: str) -> dict[str, Any]:
    path = _batch_manifest_path(settings, _validate_id(batch_id, "æ‰¹æ¬¡"))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="æ‰¹æ¬¡ä¸å­˜åœ¨")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="æ‰¹æ¬¡è®°å½•æŸå") from error
    if payload.get("batch_id") != batch_id or not isinstance(payload.get("task_ids"), list):
        raise HTTPException(status_code=500, detail="æ‰¹æ¬¡è®°å½•æ— æ•ˆ")
    return payload


def _media_stem(task: dict[str, Any]) -> str:
    source_name = Path(task.get("original_filename") or "task").stem
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source_name).strip(" .")
    return safe_name or "task"


def _export_filename(task: dict[str, Any], internal_name: str) -> str:
    suffix = ".srt" if internal_name == "subtitles.srt" else ".txt"
    return f"{_media_stem(task)}{suffix}"


def _selected_export_files(
    types: list[ExportType] | None,
    legacy_kind: LegacyExportKind | None,
) -> tuple[str, ...]:
    if types:
        selected_types = set(types)
        return tuple(
            filename
            for export_type, filename in EXPORT_FILES_BY_TYPE.items()
            if export_type in selected_types
        )
    if legacy_kind is not None:
        return EXPORT_FILES_BY_KIND[legacy_kind]
    return (EXPORT_FILES_BY_TYPE["text"],)


class WebTaskRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._items: dict[str, dict[str, Any]] = {}

    def set(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._items[task_id] = payload

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(task_id)
            return dict(item) if item is not None else None

    def remove(self, task_id: str) -> None:
        with self._lock:
            self._items.pop(task_id, None)


TaskRunner = Callable[[dict[str, Any], dict[str, Any]], str | None]


def _run_pipeline_job(
    settings_payload: dict[str, Any],
    job: dict[str, Any],
) -> str | None:
    """Run one persisted job in an isolated worker process."""
    settings = Settings.model_validate(settings_payload)
    subtitle_value = job.get("subtitle_path")
    subtitle_path = Path(str(subtitle_value)) if subtitle_value else None
    try:
        TranscriptionPipeline(settings).run(
            Path(str(job["media_path"])),
            output_dir=settings.paths.output_dir / str(job["task_id"]),
            mode=FusionMode(str(job["mode"])),
            external_subtitle=subtitle_path,
            task_id=str(job["task_id"]),
            original_filename=str(job["original_filename"]),
            batch_id=str(job["batch_id"]) if job.get("batch_id") else None,
            hard_subtitles=bool(job.get("hard_subtitles")),
        )
    except Exception as error:
        return str(error)
    return None


async def _save_upload(upload: UploadFile, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with target.open("wb") as handle:
        while chunk := await upload.read(CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"ä¸Šä¼ æ–‡ä»¶è¶…è¿‡ {MAX_UPLOAD_GB} GB é™åˆ¶",
                )
            handle.write(chunk)
    await upload.close()
    if total == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="ä¸Šä¼ æ–‡ä»¶ä¸ºç©º")
    return total


def _manifest_payload(manifest: TaskManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    if manifest.input_path.is_file():
        payload["media_size"] = manifest.input_path.stat().st_size
    payload["download_files"] = [
        name for name in sorted(DOWNLOAD_FILES) if (manifest.output_dir / name).is_file()
    ]
    transcript = manifest.output_dir / "transcript.txt"
    if transcript.is_file():
        payload["transcript_preview"] = transcript.read_text(encoding="utf-8")
    for probe_path in (
        manifest.output_dir / "probe.json",
        manifest.work_dir / "probe.json",
    ):
        if not probe_path.is_file():
            continue
        try:
            duration = json.loads(probe_path.read_text(encoding="utf-8")).get("duration")
            if isinstance(duration, int | float) and duration > 0:
                payload["media_duration"] = float(duration)
        except (OSError, ValueError):
            pass
        break
    return payload


def create_app(
    settings: Settings | None = None,
    *,
    task_executor: Executor | None = None,
    task_runner: TaskRunner = _run_pipeline_job,
) -> FastAPI:
    resolved_settings = settings or load_settings()
    static_dir = Path(__file__).with_name("static")
    multipart_temp_dir = resolved_settings.paths.work_dir.resolve() / "multipart"
    multipart_temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(multipart_temp_dir)
    registry = WebTaskRegistry()
    executor = task_executor or ProcessPoolExecutor(
        max_workers=1,
        mp_context=get_context("spawn"),
    )
    settings_payload = resolved_settings.model_dump(mode="json")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        recover_pending_tasks()
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Video2Txt", version=__version__, lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.registry = registry
    app.state.executor = executor
    app.state.multipart_temp_dir = multipart_temp_dir

    @app.exception_handler(StarletteHTTPException)
    async def friendly_http_error(
        _request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        detail = error.detail
        if detail == "There was an error parsing the body":
            detail = "ä¸Šä¼ è¯·æ±‚è§£æžå¤±è´¥ï¼Œè¯·æ£€æŸ¥é¡¹ç›®ç£ç›˜ç©ºé—´åŽé‡æ–°é€‰æ‹©æ–‡ä»¶ä¸Šä¼ "
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": detail},
            headers=error.headers,
        )

    def load_task_manifest(task_id: str) -> TaskManifest | None:
        safe_task_id = _validate_id(task_id, "ä»»åŠ¡")
        for root in (
            resolved_settings.paths.output_dir.resolve(),
            resolved_settings.paths.work_dir.resolve(),
        ):
            manifest_path = root / safe_task_id / "task.json"
            if manifest_path.is_file():
                return TaskManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
        return None

    def task_payload(task_id: str) -> dict[str, Any]:
        safe_task_id = _validate_id(task_id, "ä»»åŠ¡")
        manifest = load_task_manifest(safe_task_id)
        if manifest is not None:
            return _manifest_payload(manifest)
        queued = registry.get(safe_task_id)
        if queued is not None:
            return queued
        raise HTTPException(status_code=404, detail="ä»»åŠ¡ä¸å­˜åœ¨")

    def persist_job(
        queued: dict[str, Any],
        media_path: Path,
        subtitle_path: Path | None,
        mode: FusionMode,
    ) -> dict[str, Any]:
        job = {
            **queued,
            "media_path": str(media_path.resolve()),
            "subtitle_path": str(subtitle_path.resolve())
            if subtitle_path is not None
            else None,
            "mode": mode.value,
        }
        _write_json_atomic(
            _task_job_path(resolved_settings, queued["task_id"]),
            job,
        )
        return job

    def submit_task(
        queued: dict[str, Any],
        media_path: Path,
        subtitle_path: Path | None,
        mode: FusionMode,
    ) -> None:
        task_id = queued["task_id"]
        registry.set(task_id, queued)
        job = persist_job(queued, media_path, subtitle_path, mode)

        def finalize_job(future: Future[str | None]) -> None:
            worker_crashed = False
            try:
                error_message = future.result()
            except Exception as error:
                worker_crashed = True
                error_message = f"å·¥ä½œè¿›ç¨‹å¼‚å¸¸é€€å‡ºï¼š{error}"
            if error_message:
                registry.set(
                    task_id,
                    {**queued, "status": "failed", "error": error_message},
                )
            if not worker_crashed:
                _task_job_path(resolved_settings, task_id).unlink(missing_ok=True)

        future = executor.submit(task_runner, settings_payload, job)
        future.add_done_callback(finalize_job)

    def recover_pending_tasks() -> None:
        queue_dir = resolved_settings.paths.work_dir.resolve() / "queue"
        if not queue_dir.is_dir():
            return
        for job_path in sorted(queue_dir.glob("*.json")):
            try:
                job = json.loads(job_path.read_text(encoding="utf-8"))
                task_id = _validate_id(str(job["task_id"]), "ä»»åŠ¡")
                manifest = load_task_manifest(task_id)
                if manifest is not None and manifest.status in TERMINAL_STATUSES:
                    job_path.unlink(missing_ok=True)
                    continue
                media_path = Path(str(job["media_path"]))
                subtitle_value = job.get("subtitle_path")
                subtitle_path = Path(str(subtitle_value)) if subtitle_value else None
                if not media_path.is_file() or (
                    subtitle_path is not None and not subtitle_path.is_file()
                ):
                    registry.set(
                        task_id,
                        {
                            **job,
                            "status": "failed",
                            "error": "æ¢å¤ä»»åŠ¡å¤±è´¥ï¼šåŽŸå§‹ä¸Šä¼ æ–‡ä»¶ä¸å­˜åœ¨",
                        },
                    )
                    job_path.unlink(missing_ok=True)
                    continue
                queued = {
                    key: job.get(key)
                    for key in (
                        "task_id",
                        "batch_id",
                        "status",
                        "mode",
                        "hard_subtitles",
                        "original_filename",
                        "media_size",
                        "warnings",Û_6¶‰žËkºwµç@€€€€€€€€€€€¤(€€€€€€€€€€€€€€€•á•ÁÐ€¡=MÉÉ½È°Y…±Õ•ÉÉ½È¤è(€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€€€€€µ…¹¥™•ÍÑÍmµ…¹¥™•ÍÐ¹Ñ…Í­}¥‘t€ôµ…¹¥™•ÍÐ(€€€€€€€¥˜…¹ä¡µ…¹¥™•ÍÐ¹ÍÑ…ÑÕÌ¹½Ð¥¸QI5%91}MQQUML™½Èµ…¹¥™•ÍÐ¥¸µ…¹¥™•ÍÑÌ¹Ù…±Õ•Ì ¤¤è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹šr'’îï–*‡š¶–r£¢þC¢†3¾ò3šj’â7¢÷šâž¦ë–£¦£’îï–*„ˆ¤((€€€€€€€™É••‘}‰åÑ•Ì€ô€À(€€€€€€€™½È¡¥±¥¸±¥ÍÐ¡½ÕÑÁÕÑ}É½½Ð¹¥Ñ•É‘¥È ¤¤¥˜½ÕÑÁÕÑ}É½½Ð¹¥Í}‘¥È ¤•±Í”mtè(€€€€€€€€€€€™É••‘}‰åÑ•Ì€¬ô}É•µ½Ù•}¡¥±‘}Á…Ñ ¡½ÕÑÁÕÑ}É½½Ð°¡¥±¤(€€€€€€€™½Èµ…¹¥™•ÍÐ¥¸µ…¹¥™•ÍÑÌ¹Ù…±Õ•Ì ¤è(€€€€€€€€€€€™É••‘}‰åÑ•Ì€¬ô}É•µ½Ù•}¡¥±‘}Á…Ñ ¡Ý½É­}É½½Ð°Ý½É­}É½½Ð€¼µ…¹¥™•ÍÐ¹Ñ…Í­}¥¤(€€€€€€€€€€€É•¥ÍÑÉä¹É•µ½Ù”¡µ…¹¥™•ÍÐ¹Ñ…Í­}¥¤(€€€€€€€™½Èµ…¹…•‘}¹…µ”¥¸€ ‰ÕÁ±½…‘Ìˆ°€‰‰…Ñ¡•Ìˆ¤è(€€€€€€€€€€€µ…¹…•‘}É½½Ð€ôÝ½É­}É½½Ð€¼µ…¹…•‘}¹…µ”(€€€€€€€€€€€¥˜¹½Ðµ…¹…•‘}É½½Ð¹¥Í}‘¥È ¤è(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€™½È¡¥±¥¸±¥ÍÐ¡µ…¹…•‘}É½½Ð¹¥Ñ•É‘¥È ¤¤è(€€€€€€€€€€€€€€€™É••‘}‰åÑ•Ì€¬ô}É•µ½Ù•}¡¥±‘}Á…Ñ ¡µ…¹…•‘}É½½Ð°¡¥±¤((€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€‰±•…É•‘}Ñ…Í­Ìˆè±•¸¡µ…¹¥™•ÍÑÌ¤°(€€€€€€€€€€€€‰™É••‘}‰åÑ•Ìˆè™É••‘}‰åÑ•Ì°(€€€€€€€€€€€€¨©ÍÑ½É…•}Á…å±½… ¤°(€€€€€€€ô((€€€…ÁÀ¹•Ð ˆ½…Á¤½Ñ…Í­Ì½íÑ…Í­}¥‘ôˆ¤(€€€‘•˜•Ñ}Ñ…Í¬¡Ñ…Í­}¥èÍÑÈ¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€€€€€É•ÑÕÉ¸Ñ…Í­}Á…å±½…¡Ñ…Í­}¥¤((€€€…ÁÀ¹Á½ÍÐ ˆ½…Á¤½Ñ…Í­Ì½íÑ…Í­}¥‘ô½É•ÑÉäˆ°ÍÑ…ÑÕÍ}½‘”ôÈÀÈ¤(€€€‘•˜É•ÑÉå}Ñ…Í¬¡Ñ…Í­}¥èÍÑÈ¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€€€€€µ…¹¥™•ÍÐ€ô±½…‘}Ñ…Í­}µ…¹¥™•ÍÐ¡Ñ…Í­}¥¤(€€€€€€€¥˜µ…¹¥™•ÍÐ¥Ì9½¹”è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°ô‹’îï–*‡’â7–¶c–r ˆ¤(€€€€€€€¥˜µ…¹¥™•ÍÐ¹ÍÑ…ÑÕÌ€„ôQ…Í­MÑ…ÑÕÌ¹%1è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹–>«šr'–’Ç¢Ò—’îï–*‡–>¿’î—¦7¢¾Tˆ¤(€€€€€€€¥˜¹½Ðµ…¹¥™•ÍÐ¹¥¹ÁÕÑ}Á…Ñ ¹¥Í}™¥±” ¤è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹–:–ž/’â+’òƒšZ’îÛ–ÞË¢Š¯šâžB¾ò3š^ƒšÎW¦7¢¾Tˆ¤((€€€€€€€‰…Ñ¡}¥€ôÕÕ¥Ð ¤¹¡•à(€€€€€€€¹•Ý}Ñ…Í­}¥€ôÕÕ¥Ð ¤¹¡•à(€€€€€€€ÕÁ±½…‘}‘¥È€ô€ (€€€€€€€€€€€É•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹Ý½É­}‘¥È¹É•Í½±Ù” ¤(€€€€€€€€€€€€¼€‰ÕÁ±½…‘Ìˆ(€€€€€€€€€€€€¼‰…Ñ¡}¥(€€€€€€€€€€€€¼¹•Ý}Ñ…Í­}¥(€€€€€€€€¤(€€€€€€€µ•‘¥…}Á…Ñ €ôÕÁ±½…‘}‘¥È€¼˜‰Í½ÕÉ•íµ…¹¥™•ÍÐ¹¥¹ÁÕÑ}Á…Ñ ¹ÍÕ™™¥à¹±½Ý•È ¥ôˆ(€€€€€€€ÍÕ‰Ñ¥Ñ±•}Ù…±Õ”€ôµ…¹¥™•ÍÐ¹…ÉÑ¥™…ÑÌ¹•Ð ‰ÍÕ‰Ñ¥Ñ±•}Í½ÕÉ”ˆ¤(€€€€€€€ÍÕ‰Ñ¥Ñ±•}Í½ÕÉ”€ôA…Ñ ¡ÍÕ‰Ñ¥Ñ±•}Ù…±Õ”¤¥˜ÍÕ‰Ñ¥Ñ±•}Ù…±Õ”•±Í”9½¹”(€€€€€€€ÍÕ‰Ñ¥Ñ±•}Á…Ñ èA…Ñ ð9½¹”€ô9½¹”(€€€€€€€ÑÉäè(€€€€€€€€€€€ÕÁ±½…‘}‘¥È¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤(€€€€€€€€€€€Í¡ÕÑ¥°¹½ÁäÈ¡µ…¹¥™•ÍÐ¹¥¹ÁÕÑ}Á…Ñ °µ•‘¥…}Á…Ñ ¤(€€€€€€€€€€€¥˜ÍÕ‰Ñ¥Ñ±•}Í½ÕÉ”¥Ì¹½Ð9½¹”…¹ÍÕ‰Ñ¥Ñ±•}Í½ÕÉ”¹¥Í}™¥±” ¤è(€€€€€€€€€€€€€€€ÍÕ‰Ñ¥Ñ±•}Á…Ñ €ôÕÁ±½…‘}‘¥È€¼˜‰ÍÕ‰Ñ¥Ñ±•íÍÕ‰Ñ¥Ñ±•}Í½ÕÉ”¹ÍÕ™™¥à¹±½Ý•È ¥ôˆ(€€€€€€€€€€€€€€€Í¡ÕÑ¥°¹½ÁäÈ¡ÍÕ‰Ñ¥Ñ±•}Í½ÕÉ”°ÍÕ‰Ñ¥Ñ±•}Á…Ñ ¤(€€€€€€€•á•ÁÐá•ÁÑ¥½¸è(€€€€€€€€€€€}É•µ½Ù•}¡¥±‘}Á…Ñ  (€€€€€€€€€€€€€€€É•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹Ý½É­}‘¥È¹É•Í½±Ù” ¤°(€€€€€€€€€€€€€€€É•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹Ý½É­}‘¥È¹É•Í½±Ù” ¤€¼€‰ÕÁ±½…‘Ìˆ€¼‰…Ñ¡}¥°(€€€€€€€€€€€€¤(€€€€€€€€€€€É…¥Í”((€€€€€€€ÅÕ•Õ•€ôì(€€€€€€€€€€€€‰Ñ…Í­}¥ˆè¹•Ý}Ñ…Í­}¥°(€€€€€€€€€€€€‰‰…Ñ¡}¥ˆè‰…Ñ¡}¥°(€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰ÅÕ•Õ•ˆ°(€€€€€€€€€€€€‰µ½‘”ˆèµ…¹¥™•ÍÐ¹µ½‘”¹Ù…±Õ”°(€€€€€€€€€€€€‰¡…É‘}ÍÕ‰Ñ¥Ñ±•Ìˆèµ…¹¥™•ÍÐ¹¡…É‘}ÍÕ‰Ñ¥Ñ±•Ì°(€€€€€€€€€€€€‰½É¥¥¹…±}™¥±•¹…µ”ˆèµ…¹¥™•ÍÐ¹½É¥¥¹…±}™¥±•¹…µ”(€€€€€€€€€€€½Èµ…¹¥™•ÍÐ¹¥¹ÁÕÑ}Á…Ñ ¹¹…µ”°(€€€€€€€€€€€€‰µ•‘¥…}Í¥é”ˆèµ•‘¥…}Á…Ñ ¹ÍÑ…Ð ¤¹ÍÑ}Í¥é”°(€€€€€€€€€€€€‰Ý…É¹¥¹Ìˆèmt°(€€€€€€€€€€€€‰•ÉÉ½Èˆè9½¹”°(€€€€€€€ô(€€€€€€€‰…Ñ €ôì(€€€€€€€€€€€€‰‰…Ñ¡}¥ˆè‰…Ñ¡}¥°(€€€€€€€€€€€€‰µ½‘”ˆèµ…¹¥™•ÍÐ¹µ½‘”¹Ù…±Õ”°(€€€€€€€€€€€€‰¡…É‘}ÍÕ‰Ñ¥Ñ±•Ìˆèµ…¹¥™•ÍÐ¹¡…É‘}ÍÕ‰Ñ¥Ñ±•Ì°(€€€€€€€€€€€€‰É•…Ñ•‘}…Ðˆè‘…Ñ•Ñ¥µ”¹¹½Ü ¤¹…ÍÑ¥µ•é½¹” ¤¹¥Í½™½Éµ…Ð¡Ñ¥µ•ÍÁ•Œô‰Í•½¹‘Ìˆ¤°(€€€€€€€€€€€€‰Ñ…Í­}¥‘Ìˆèm¹•Ý}Ñ…Í­}¥‘t°(€€€€€€€ô(€€€€€€€}ÝÉ¥Ñ•}‰…Ñ¡}µ…¹¥™•ÍÐ¡É•Í½±Ù•‘}Í•ÑÑ¥¹Ì°‰…Ñ ¤(€€€€€€€ÍÕ‰µ¥Ñ}Ñ…Í¬¡ÅÕ•Õ•°µ•‘¥…}Á…Ñ °ÍÕ‰Ñ¥Ñ±•}Á…Ñ °µ…¹¥™•ÍÐ¹µ½‘”¤(€€€€€€€É•ÑÕÉ¸ì¨©‰…Ñ °€‰Ñ…Í­ÌˆèmÅÕ•Õ•‘uô((€€€…ÁÀ¹‘•±•Ñ” ˆ½…Á¤½Ñ…Í­Ì½íÑ…Í­}¥‘ôˆ¤(€€€‘•˜‘•±•Ñ•}Ñ…Í¬¡Ñ…Í­}¥èÍÑÈ¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€€€€€Á…å±½…€ôÑ…Í­}Á…å±½…¡Ñ…Í­}¥¤(€€€€€€€ÑÉäè(€€€€€€€€€€€ÍÑ…ÑÕÌ€ôQ…Í­MÑ…ÑÕÌ¡Á…å±½…‘l‰ÍÑ…ÑÕÌ‰t¤(€€€€€€€•á•ÁÐY…±Õ•ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹’îï–*‡ž*Ûšš^ƒšV ˆ¤™É½´•ÉÉ½È(€€€€€€€¥˜ÍÑ…ÑÕÌ¹½Ð¥¸QI5%91}MQQUMLè(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹¢þC¢†3’â·žj’îï–*‡’â7¢÷–"ƒ¦fˆ¤((€€€€€€€Í…™•}Ñ…Í­}¥€ô}Ù…±¥‘…Ñ•}¥¡Ñ…Í­}¥°€‹’îï–*„ˆ¤(€€€€€€€Ý½É­}É½½Ð€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹Ý½É­}‘¥È¹É•Í½±Ù” ¤(€€€€€€€½ÕÑÁÕÑ}É½½Ð€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹½ÕÑÁÕÑ}‘¥È¹É•Í½±Ù” ¤(€€€€€€€‰…Ñ¡}¥€ôÁ…å±½…¹•Ð ‰‰…Ñ¡}¥ˆ¤(€€€€€€€™É••‘}‰åÑ•Ì€ô€À(€€€€€€€™½ÈÑ…É•Ð°É½½Ð¥¸€ (€€€€€€€€€€€€¡Ý½É­}É½½Ð€¼Í…™•}Ñ…Í­}¥°Ý½É­}É½½Ð¤°(€€€€€€€€€€€€¡½ÕÑÁÕÑ}É½½Ð€¼Í…™•}Ñ…Í­}¥°½ÕÑÁÕÑ}É½½Ð¤°(€€€€€€€€¤è(€€€€€€€€€€€™É••‘}‰åÑ•Ì€¬ô}É•µ½Ù•}¡¥±‘}Á…Ñ ¡É½½Ð°Ñ…É•Ð¤(€€€€€€€ÕÁ±½…‘}É½ÕÀ€ôÍÑÈ¡‰…Ñ¡}¥½ÈÍ…™•}Ñ…Í­}¥¤(€€€€€€€™É••‘}‰åÑ•Ì€¬ô}É•µ½Ù•}¡¥±‘}Á…Ñ  (€€€€€€€€€€€Ý½É­}É½½Ð°(€€€€€€€€€€€Ý½É­}É½½Ð€¼€‰ÕÁ±½…‘Ìˆ€¼ÕÁ±½…‘}É½ÕÀ€¼Í…™•}Ñ…Í­}¥°(€€€€€€€€¤(€€€€€€€}Ñ…Í­}©½‰}Á…Ñ ¡É•Í½±Ù•‘}Í•ÑÑ¥¹Ì°Í…™•}Ñ…Í­}¥¤¹Õ¹±¥¹¬¡µ¥ÍÍ¥¹}½¬õQÉÕ”¤(€€€€€€€É•¥ÍÑÉä¹É•µ½Ù”¡Í…™•}Ñ…Í­}¥¤((€€€€€€€¥˜‰…Ñ¡}¥è(€€€€€€€€€€€‰…Ñ¡}Á…Ñ €ô}‰…Ñ¡}µ…¹¥™•ÍÑ}Á…Ñ  (€€€€€€€€€€€€€€€É•Í½±Ù•‘}Í•ÑÑ¥¹Ì°}Ù…±¥‘…Ñ•}¥¡ÍÑÈ¡‰…Ñ¡}¥¤°€‹š&çš²„ˆ¤(€€€€€€€€€€€€¤(€€€€€€€€€€€¥˜‰…Ñ¡}Á…Ñ ¹¥Í}™¥±” ¤è(€€€€€€€€€€€€€€€‰…Ñ €ô©Í½¸¹±½…‘Ì¡‰…Ñ¡}Á…Ñ ¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤¤(€€€€€€€€€€€€€€€‰…Ñ¡l‰Ñ…Í­}¥‘Ì‰t€ôl(€€€€€€€€€€€€€€€€€€€¥Ñ•´™½È¥Ñ•´¥¸‰…Ñ ¹•Ð ‰Ñ…Í­}¥‘Ìˆ°mt¤¥˜¥Ñ•´€„ôÍ…™•}Ñ…Í­}¥(€€€€€€€€€€€€€€€t(€€€€€€€€€€€€€€€¥˜‰…Ñ¡l‰Ñ…Í­}¥‘Ì‰tè(€€€€€€€€€€€€€€€€€€€}ÝÉ¥Ñ•}‰…Ñ¡}µ…¹¥™•ÍÐ¡É•Í½±Ù•‘}Í•ÑÑ¥¹Ì°‰…Ñ ¤(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€‰…Ñ¡}Á…Ñ ¹Õ¹±¥¹¬¡µ¥ÍÍ¥¹}½¬õQÉÕ”¤(€€€€€€€É•ÑÕÉ¸ì‰Ñ…Í­}¥ˆèÍ…™•}Ñ…Í­}¥°€‰™É••‘}‰åÑ•Ìˆè™É••‘}‰åÑ•Ì°€¨©ÍÑ½É…•}Á…å±½… ¥ô((€€€…ÁÀ¹•Ð ˆ½…Á¤½Ñ…Í­Ì½íÑ…Í­}¥‘ô½™¥±•Ì½í™¥±•¹…µ•ôˆ¤(€€€‘•˜‘½Ý¹±½…‘}Ñ…Í­}™¥±”¡Ñ…Í­}¥èÍÑÈ°™¥±•¹…µ”èÍÑÈ¤€´ø¥±•I•ÍÁ½¹Í”è(€€€€€€€¥˜™¥±•¹…µ”¹½Ð¥¸=]91=}%1Lè(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°ô‹šZ’îÛ’â7–¶c–r ˆ¤(€€€€€€€Í…™•}Ñ…Í­}¥€ô}Ù…±¥‘…Ñ•}¥¡Ñ…Í­}¥°€‹’îï–*„ˆ¤(€€€€€€€Ñ…É•Ð€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹½ÕÑÁÕÑ}‘¥È¹É•Í½±Ù” ¤€¼Í…™•}Ñ…Í­}¥€¼™¥±•¹…µ”(€€€€€€€¥˜¹½ÐÑ…É•Ð¹¥Í}™¥±” ¤è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°ô‹šZ’îÛ’â7–¶c–r ˆ¤(€€€€€€€Ñ…Í¬€ôÑ…Í­}Á…å±½…¡Í…™•}Ñ…Í­}¥¤(€€€€€€€É•ÑÕÉ¸¥±•I•ÍÁ½¹Í”¡Ñ…É•Ð°™¥±•¹…µ”õ}•áÁ½ÉÑ}™¥±•¹…µ”¡Ñ…Í¬°™¥±•¹…µ”¤¤((€€€…ÁÀ¹•Ð ˆ½…Á¤½Ñ…Í­Ì½íÑ…Í­}¥‘ô½•áÁ½ÉÐˆ¤(€€€‘•˜•áÁ½ÉÑ}Ñ…Í¬ (€€€€€€€Ñ…Í­}¥èÍÑÈ°(€€€€€€€ÑåÁ•Ìè¹¹½Ñ…Ñ•‘m±¥ÍÑmáÁ½ÉÑQåÁ•tð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€€€€€­¥¹è1•…åáÁ½ÉÑ-¥¹ð9½¹”€ô9½¹”°(€€€€¤€´ø¥±•I•ÍÁ½¹Í”è(€€€€€€€Ñ…Í¬€ôÑ…Í­}Á…å±½…¡Ñ…Í­}¥¤(€€€€€€€¥˜Ñ…Í­l‰ÍÑ…ÑÕÌ‰t€„ô€‰½µÁ±•Ñ•ˆè(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹’îï–*‡–Âkšr«–º3š"@ˆ¤(€€€€€€€½ÕÑÁÕÑ}‘¥È€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹½ÕÑÁÕÑ}‘¥È¹É•Í½±Ù” ¤€¼Ñ…Í­l‰Ñ…Í­}¥‰t(€€€€€€€Í•±•Ñ•€ô}Í•±•Ñ•‘}•áÁ½ÉÑ}™¥±•Ì¡ÑåÁ•Ì°­¥¹¤(€€€€€€€…Ù…¥±…‰±”€ôm¹…µ”™½È¹…µ”¥¸Í•±•Ñ•¥˜€¡½ÕÑÁÕÑ}‘¥È€¼¹…µ”¤¹¥Í}™¥±” ¥t(€€€€€€€¥˜±•¸¡…Ù…¥±…‰±”¤€„ô±•¸¡Í•±•Ñ•¤è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°ô‹š&¦'žÆï–z/žjšZ’îÛ’â7–¶c–r ˆ¤(€€€€€€€¥˜¹½Ð…Ù…¥±…‰±”è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°ô‹šÊ‡šr'–>¿’â/¢ö÷žjšZ’îØˆ¤(€€€€€€€¥˜±•¸¡Í•±•Ñ•¤€ôô€Äè(€€€€€€€€€€€¥¹Ñ•É¹…±}¹…µ”€ô…Ù…¥±…‰±•lÁt(€€€€€€€€€€€É•ÑÕÉ¸¥±•I•ÍÁ½¹Í” (€€€€€€€€€€€€€€€½ÕÑÁÕÑ}‘¥È€¼¥¹Ñ•É¹…±}¹…µ”°(€€€€€€€€€€€€€€€™¥±•¹…µ”õ}•áÁ½ÉÑ}™¥±•¹…µ”¡Ñ…Í¬°¥¹Ñ•É¹…±}¹…µ”¤°(€€€€€€€€€€€€¤((€€€€€€€•áÁ½ÉÑ}‘¥È€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹Ý½É­}‘¥È¹É•Í½±Ù” ¤€¼€‰•áÁ½ÉÑÌˆ(€€€€€€€•áÁ½ÉÑ}‘¥È¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤(€€€€€€€…É¡¥Ù•}Á…Ñ €ô•áÁ½ÉÑ}‘¥È€¼˜‰íÑ…Í­lÑ…Í­}¥uôµíÕÕ¥Ð ¤¹¡•áô¹é¥Àˆ(€€€€€€€Ý¥Ñ é¥Á™¥±”¹i¥Á¥±” (€€€€€€€€€€€…É¡¥Ù•}Á…Ñ °€‰Üˆ°½µÁÉ•ÍÍ¥½¸õé¥Á™¥±”¹i%A}1Q(€€€€€€€€¤…Ì…É¡¥Ù”è(€€€€€€€€€€€™½È¥¹Ñ•É¹…±}¹…µ”¥¸…Ù…¥±…‰±”è(€€€€€€€€€€€€€€€…É¡¥Ù”¹ÝÉ¥Ñ” (€€€€€€€€€€€€€€€€€€€½ÕÑÁÕÑ}‘¥È€¼¥¹Ñ•É¹…±}¹…µ”°(€€€€€€€€€€€€€€€€€€€}•áÁ½ÉÑ}™¥±•¹…µ”¡Ñ…Í¬°¥¹Ñ•É¹…±}¹…µ”¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€É•ÑÕÉ¸¥±•I•ÍÁ½¹Í” (€€€€€€€€€€€…É¡¥Ù•}Á…Ñ °(€€€€€€€€€€€µ•‘¥…}ÑåÁ”ô‰…ÁÁ±¥…Ñ¥½¸½é¥Àˆ°(€€€€€€€€€€€™¥±•¹…µ”õ˜‰í}µ•‘¥…}ÍÑ•´¡Ñ…Í¬¥ô¹é¥Àˆ°(€€€€€€€€€€€‰…­É½Õ¹õ	…­É½Õ¹‘Q…Í¬¡…É¡¥Ù•}Á…Ñ ¹Õ¹±¥¹¬°µ¥ÍÍ¥¹}½¬õQÉÕ”¤°(€€€€€€€€¤((€€€…ÁÀ¹Á½ÍÐ ˆ½…Á¤½Ñ…Í­Ìˆ°ÍÑ…ÑÕÍ}½‘”ôÈÀÈ¤(€€€…Íå¹Œ‘•˜É•…Ñ•}Ñ…Í¬ (€€€€€€€µ•‘¥„è¹¹½Ñ…Ñ•‘mUÁ±½…‘¥±”°¥±” ¥t°(€€€€€€€ÍÕ‰Ñ¥Ñ±”è¹¹½Ñ…Ñ•‘mUÁ±½…‘¥±”ð9½¹”°¥±” ¥t€ô9½¹”°(€€€€€€€µ½‘”è¹¹½Ñ…Ñ•‘mÕÍ¥½¹5½‘”°½É´ ¥t€ôÕÍ¥½¹5½‘”¹YI	Q%4°(€€€€€€€¡…É‘}ÍÕ‰Ñ¥Ñ±•Ìè¹¹½Ñ…Ñ•‘m‰½½°°½É´ ¥t€ô…±Í”°(€€€€¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€€€€€µ½‘•±}Á…Ñ €ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹…ÍÈ¹µ½‘•±}Á…Ñ (€€€€€€€¥˜µ½‘•±}Á…Ñ ¥Ì9½¹”½È¹½Ðµ½‘•±}Á…Ñ ¹¥Í}‘¥È ¤è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÔÀÌ°‘•Ñ…¥°ô‹šr³–rÀMHƒš¢‡–z/–Âkšr«¦7žö¸ˆ¤((€€€€€€€ÅÕ•Õ•°µ•‘¥…}Á…Ñ °ÍÕ‰Ñ¥Ñ±•}Á…Ñ €ô…Ý…¥ÐÁÉ•Á…É•}Ñ…Í¬ (€€€€€€€€€€€µ•‘¥„°ÍÕ‰Ñ¥Ñ±”°µ½‘”°¡…É‘}ÍÕ‰Ñ¥Ñ±•Ìõ¡…É‘}ÍÕ‰Ñ¥Ñ±•Ì(€€€€€€€€¤(€€€€€€€ÍÕ‰µ¥Ñ}Ñ…Í¬¡ÅÕ•Õ•°µ•‘¥…}Á…Ñ °ÍÕ‰Ñ¥Ñ±•}Á…Ñ °µ½‘”¤(€€€€€€€É•ÑÕÉ¸ÅÕ•Õ•((€€€…ÁÀ¹Á½ÍÐ ˆ½…Á¤½‰…Ñ¡•Ìˆ°ÍÑ…ÑÕÍ}½‘”ôÈÀÈ¤(€€€…Íå¹Œ‘•˜É•…Ñ•}‰…Ñ  (€€€€€€€µ•‘¥„è¹¹½Ñ…Ñ•‘m±¥ÍÑmUÁ±½…‘¥±•t°¥±” ¥t°(€€€€€€€ÍÕ‰Ñ¥Ñ±•Ìè¹¹½Ñ…Ñ•‘m±¥ÍÑmUÁ±½…‘¥±”ðÍÑÉtð9½¹”°¥±” ¥t€ô9½¹”°(€€€€€€€µ½‘”è¹¹½Ñ…Ñ•‘mÕÍ¥½¹5½‘”°½É´ ¥t€ôÕÍ¥½¹5½‘”¹YI	Q%4°(€€€€€€€¡…É‘}ÍÕ‰Ñ¥Ñ±•Ìè¹¹½Ñ…Ñ•‘m‰½½°°½É´ ¥t€ô…±Í”°(€€€€¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€€€€€µ½‘•±}Á…Ñ €ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹…ÍÈ¹µ½‘•±}Á…Ñ (€€€€€€€¥˜µ½‘•±}Á…Ñ ¥Ì9½¹”½È¹½Ðµ½‘•±}Á…Ñ ¹¥Í}‘¥È ¤è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÔÀÌ°‘•Ñ…¥°ô‹šr³–rÀMHƒš¢‡–z/–Âkšr«¦7žö¸ˆ¤(€€€€€€€¥˜¹½Ðµ•‘¥„è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÀ°‘•Ñ…¥°ô‹¢¾ß¢Ï–ÂG’â+’òƒ’â’â«–ªK’öOšZ’îØˆ¤(€€€€€€€¥˜±•¸¡µ•‘¥„¤€ø5a}	Q!}%1Lè(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÀ°‘•Ñ…¥°õ˜‹–6Wš&çšr–’hí5a}	Q!}%1Môƒ’â«–ªK’öOšZ’îØˆ¤(€€€€€€€ÍÕ‰Ñ¥Ñ±•}ÕÁ±½…‘Ìè±¥ÍÑmUÁ±½…‘¥±•t€ômt(€€€€€€€™½ÈÕÁ±½…¥¸ÍÕ‰Ñ¥Ñ±•Ì½Èmtè(€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡ÕÁ±½…°ÍÑÈ¤è(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€¥˜ÕÁ±½…¹™¥±•¹…µ”è(€€€€€€€€€€€€€€€ÍÕ‰Ñ¥Ñ±•}ÕÁ±½…‘Ì¹…ÁÁ•¹¡ÕÁ±½…¤(€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€…Ý…¥ÐÕÁ±½…¹±½Í” ¤(€€€€€€€¥˜±•¸¡ÍÕ‰Ñ¥Ñ±•}ÕÁ±½…‘Ì¤€ø5a}	Q!}%1Lè(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÀ°‘•Ñ…¥°õ˜‹–6Wš&çšr–’hí5a}	Q!}%1Môƒ’â«–¶_–æWšZ’îØˆ¤((€€€€€€€™½ÈÕÁ±½…¥¸µ•‘¥„è(€€€€€€€€€€€¥˜A…Ñ ¡ÕÁ±½…¹™¥±•¹…µ”½È€ˆˆ¤¹ÍÕ™™¥à¹±½Ý•È ¤¹½Ð¥¸5%}aQ9M%=9Lè(€€€€€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÀ°‘•Ñ…¥°õ˜‹’â7šR¿š2–ªK’öOšZ’îÛ¾òiíÕÁ±½…¹™¥±•¹…µ•ôˆ¤(€€€€€€€ÍÕ‰Ñ¥Ñ±•}µ…Àè‘¥ÑmÍÑÈ°UÁ±½…‘¥±•t€ôíô(€€€€€€€™½ÈÕÁ±½…¥¸ÍÕ‰Ñ¥Ñ±•}ÕÁ±½…‘Ìè(€€€€€€€€€€€¥˜A…Ñ ¡ÕÁ±½…¹™¥±•¹…µ”½È€ˆˆ¤¹ÍÕ™™¥à¹±½Ý•È ¤¹½Ð¥¸MU	Q%Q1}aQ9M%=9Lè(€€€€€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÀ°‘•Ñ…¥°õ˜‹’â7šR¿š2–¶_–æWšZ’îÛ¾òiíÕÁ±½…¹™¥±•¹…µ•ôˆ¤(€€€€€€€€€€€­•ä€ôA…Ñ ¡ÕÁ±½…¹™¥±•¹…µ”½È€ˆˆ¤¹ÍÑ•´¹…Í•™½± ¤(€€€€€€€€€€€¥˜­•ä¥¸ÍÕ‰Ñ¥Ñ±•}µ…Àè(€€€€€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÀ°‘•Ñ…¥°õ˜‹–¶_–æWšZ’îÛ–B7¦7–’7¾òiíÕÁ±½…¹™¥±•¹…µ•ôˆ¤(€€€€€€€€€€€ÍÕ‰Ñ¥Ñ±•}µ…Ám­•åt€ôÕÁ±½…(€€€€€€€µ•‘¥…}ÍÑ•µÌ€ôíA…Ñ ¡ÕÁ±½…¹™¥±•¹…µ”½È€ˆˆ¤¹ÍÑ•´¹…Í•™½± ¤™½ÈÕÁ±½…¥¸µ•‘¥…ô(€€€€€€€Õ¹µ…Ñ¡•€ôl(€€€€€€€€€€€ÕÁ±½…¹™¥±•¹…µ”(€€€€€€€€€€€™½È­•ä°ÕÁ±½…¥¸ÍÕ‰Ñ¥Ñ±•}µ…À¹¥Ñ•µÌ ¤(€€€€€€€€€€€¥˜­•ä¹½Ð¥¸µ•‘¥…}ÍÑ•µÌ(€€€€€€€t(€€€€€€€¥˜Õ¹µ…Ñ¡•è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÀ°‘•Ñ…¥°õ˜‹–¶_–æWšÊ‡šr'–B3–B7–ªK’öO¾òiíÕ¹µ…Ñ¡•‘lÁuôˆ¤((€€€€€€€‰…Ñ¡}¥€ôÕÕ¥Ð ¤¹¡•à(€€€€€€€ÁÉ•Á…É•è±¥ÍÑmÑÕÁ±•m‘¥ÑmÍÑÈ°¹åt°A…Ñ °A…Ñ ð9½¹•ut€ômt(€€€€€€€ÑÉäè(€€€€€€€€€€€™½ÈÕÁ±½…¥¸µ•‘¥„è(€€€€€€€€€€€€€€€ÍÕ‰Ñ¥Ñ±”€ôÍÕ‰Ñ¥Ñ±•}µ…À¹•Ð¡A…Ñ ¡ÕÁ±½…¹™¥±•¹…µ”½È€ˆˆ¤¹ÍÑ•´¹…Í•™½± ¤¤(€€€€€€€€€€€€€€€ÁÉ•Á…É•¹…ÁÁ•¹ (€€€€€€€€€€€€€€€€€€€…Ý…¥ÐÁÉ•Á…É•}Ñ…Í¬ (€€€€€€€€€€€€€€€€€€€€€€€ÕÁ±½…°(€€€€€€€€€€€€€€€€€€€€€€€ÍÕ‰Ñ¥Ñ±”°(€€€€€€€€€€€€€€€€€€€€€€€µ½‘”°(€€€€€€€€€€€€€€€€€€€€€€€‰…Ñ¡}¥°(€€€€€€€€€€€€€€€€€€€€€€€¡…É‘}ÍÕ‰Ñ¥Ñ±•Ìõ¡…É‘}ÍÕ‰Ñ¥Ñ±•Ì°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€¤(€€€€€€€•á•ÁÐá•ÁÑ¥½¸è(€€€€€€€€€€€ÕÁ±½…‘}É½½Ð€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹Ý½É­}‘¥È¹É•Í½±Ù” ¤€¼€‰ÕÁ±½…‘Ìˆ€¼‰…Ñ¡}¥(€€€€€€€€€€€¥˜ÕÁ±½…‘}É½½Ð¹¥Í}‘¥È ¤è(€€€€€€€€€€€€€€€Í¡ÕÑ¥°¹ÉµÑÉ•”¡ÕÁ±½…‘}É½½Ð¤(€€€€€€€€€€€É…¥Í”((€€€€€€€‰…Ñ €ôì(€€€€€€€€€€€€‰‰…Ñ¡}¥ˆè‰…Ñ¡}¥°(€€€€€€€€€€€€‰µ½‘”ˆèµ½‘”¹Ù…±Õ”°(€€€€€€€€€€€€‰¡…É‘}ÍÕ‰Ñ¥Ñ±•Ìˆè¡…É‘}ÍÕ‰Ñ¥Ñ±•Ì°(€€€€€€€€€€€€‰É•…Ñ•‘}…Ðˆè‘…Ñ•Ñ¥µ”¹¹½Ü ¤¹…ÍÑ¥µ•é½¹” ¤¹¥Í½™½Éµ…Ð¡Ñ¥µ•ÍÁ•Œô‰Í•½¹‘Ìˆ¤°(€€€€€€€€€€€€‰Ñ…Í­}¥‘Ìˆèm¥Ñ•µlÁul‰Ñ…Í­}¥‰t™½È¥Ñ•´¥¸ÁÉ•Á…É•‘t°(€€€€€€€ô(€€€€€€€}ÝÉ¥Ñ•}‰…Ñ¡}µ…¹¥™•ÍÐ¡É•Í½±Ù•‘}Í•ÑÑ¥¹Ì°‰…Ñ ¤(€€€€€€€™½ÈÅÕ•Õ•°µ•‘¥…}Á…Ñ °ÍÕ‰Ñ¥Ñ±•}Á…Ñ ¥¸ÁÉ•Á…É•è(€€€€€€€€€€€ÍÕ‰µ¥Ñ}Ñ…Í¬¡ÅÕ•Õ•°µ•‘¥…}Á…Ñ °ÍÕ‰Ñ¥Ñ±•}Á…Ñ °µ½‘”¤(€€€€€€€É•ÑÕÉ¸ì¨©‰…Ñ °€‰Ñ…Í­Ìˆèm¥Ñ•µlÁt™½È¥Ñ•´¥¸ÁÉ•Á…É•‘uô((€€€…ÁÀ¹•Ð ˆ½…Á¤½‰…Ñ¡•Ì½í‰…Ñ¡}¥‘ôˆ¤(€€€‘•˜•Ñ}‰…Ñ ¡‰…Ñ¡}¥èÍÑÈ¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€€€€€‰…Ñ €ô}É•…‘}‰…Ñ¡}µ…¹¥™•ÍÐ¡É•Í½±Ù•‘}Í•ÑÑ¥¹Ì°‰…Ñ¡}¥¤(€€€€€€€É•ÑÕÉ¸ì¨©‰…Ñ °€‰Ñ…Í­ÌˆèmÑ…Í­}Á…å±½…¡Ñ…Í­}¥¤™½ÈÑ…Í­}¥¥¸‰…Ñ¡l‰Ñ…Í­}¥‘Ì‰uuô((€€€…ÁÀ¹•Ð ˆ½…Á¤½‰…Ñ¡•Ì½í‰…Ñ¡}¥‘ô½•áÁ½ÉÐ¹é¥Àˆ¤(€€€‘•˜•áÁ½ÉÑ}‰…Ñ  (€€€€€€€‰…Ñ¡}¥èÍÑÈ°(€€€€€€€ÑåÁ•Ìè¹¹½Ñ…Ñ•‘m±¥ÍÑmáÁ½ÉÑQåÁ•tð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€€€€€­¥¹è1•…åáÁ½ÉÑ-¥¹ð9½¹”€ô9½¹”°(€€€€¤€´ø¥±•I•ÍÁ½¹Í”è(€€€€€€€‰…Ñ €ô}É•…‘}‰…Ñ¡}µ…¹¥™•ÍÐ¡É•Í½±Ù•‘}Í•ÑÑ¥¹Ì°‰…Ñ¡}¥¤(€€€€€€€Ñ…Í­Ì€ômÑ…Í­}Á…å±½…¡Ñ…Í­}¥¤™½ÈÑ…Í­}¥¥¸‰…Ñ¡l‰Ñ…Í­}¥‘Ì‰ut(€€€€€€€¥˜…¹ä¡Ñ…Í­l‰ÍÑ…ÑÕÌ‰t¹½Ð¥¸ì‰½µÁ±•Ñ•ˆ°€‰™…¥±•ˆ°€‰…¹•±±•‰ô™½ÈÑ…Í¬¥¸Ñ…Í­Ì¤è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹š&çš²‡’î7–r£–’žB’â´ˆ¤(€€€€€€€½µÁ±•Ñ•€ômÑ…Í¬™½ÈÑ…Í¬¥¸Ñ…Í­Ì¥˜Ñ…Í­l‰ÍÑ…ÑÕÌ‰t€ôô€‰½µÁ±•Ñ•‰t(€€€€€€€¥˜¹½Ð½µÁ±•Ñ•è(€€€€€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀä°‘•Ñ…¥°ô‹š&çš²‡’â·šÊ‡šr'–>¿–¾ó–ëžj–º3š"C’îï–*„ˆ¤((€€€€€€€•áÁ½ÉÑ}‘¥È€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹Ý½É­}‘¥È¹É•Í½±Ù” ¤€¼€‰•áÁ½ÉÑÌˆ(€€€€€€€•áÁ½ÉÑ}‘¥È¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤(€€€€€€€…É¡¥Ù•}Á…Ñ €ô•áÁ½ÉÑ}‘¥È€¼˜‰í‰…Ñ¡}¥‘ôµíÕÕ¥Ð ¤¹¡•áô¹é¥Àˆ(€€€€€€€Í•±•Ñ•€ô}Í•±•Ñ•‘}•áÁ½ÉÑ}™¥±•Ì¡ÑåÁ•Ì°­¥¹¤(€€€€€€€Ý¥Ñ é¥Á™¥±”¹i¥Á¥±”¡…É¡¥Ù•}Á…Ñ °€‰Üˆ°½µÁÉ•ÍÍ¥½¸õé¥Á™¥±”¹i%A}1Q¤…Ì…É¡¥Ù”è(€€€€€€€€€€€ÕÍ•‘}™½±‘•ÉÌèÍ•ÑmÍÑÉt€ôÍ•Ð ¤(€€€€€€€€€€€™½ÈÑ…Í¬¥¸½µÁ±•Ñ•è(€€€€€€€€€€€€€€€½ÕÑÁÕÑ}‘¥È€ôÉ•Í½±Ù•‘}Í•ÑÑ¥¹Ì¹Á…Ñ¡Ì¹½ÕÑÁÕÑ}‘¥È¹É•Í½±Ù” ¤€¼Ñ…Í­l‰Ñ…Í­}¥‰t(€€€€€€€€€€€€€€€ÍÑ•´€ô}µ•‘¥…}ÍÑ•´¡Ñ…Í¬¤(€€€€€€€€€€€€€€€™½±‘•È€ôÍÑ•´(€€€€€€€€€€€€€€€¥˜™½±‘•È¹…Í•™½± ¤¥¸ÕÍ•‘}™½±‘•ÉÌè(€€€€€€€€€€€€€€€€€€€™½±‘•È€ô˜‰íÍÑ•µôµíÑ…Í­lÑ…Í­}¥ulèáuôˆ(€€€€€€€€€€€€€€€ÕÍ•‘}™½±‘•ÉÌ¹…‘¡™½±‘•È¹…Í•™½± ¤¤(€€€€€€€€€€€€€€€™½È¥¹Ñ•É¹…±}¹…µ”¥¸Í•±•Ñ•è(€€€€€€€€€€€€€€€€€€€Í½ÕÉ”€ô½ÕÑÁÕÑ}‘¥È€¼¥¹Ñ•É¹…±}¹…µ”(€€€€€€€€€€€€€€€€€€€¥˜Í½ÕÉ”¹¥Í}™¥±” ¤è(€€€€€€€€€€€€€€€€€€€€€€€…É¡¥Ù”¹ÝÉ¥Ñ” (€€€€€€€€€€€€€€€€€€€€€€€€€€€Í½ÕÉ”°(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰í™½±‘•Éô½í}•áÁ½ÉÑ}™¥±•¹…µ”¡Ñ…Í¬°¥¹Ñ•É¹…±}¹…µ”¥ôˆ°(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€É•ÑÕÉ¸¥±•I•ÍÁ½¹Í” (€€€€€€€€€€€…É¡¥Ù•}Á…Ñ °(€€€€€€€€€€€µ•‘¥…}ÑåÁ”ô‰…ÁÁ±¥…Ñ¥½¸½é¥Àˆ°(€€€€€€€€€€€™¥±•¹…µ”õ˜‰Ù¥‘•¼ÉÑáÐµí‰…Ñ¡}¥‘lèáuô¹é¥Àˆ°(€€€€€€€€€€€‰…­É½Õ¹õ	…­É½Õ¹‘Q…Í¬¡…É¡¥Ù•}Á…Ñ ¹Õ¹±¥¹¬°µ¥ÍÍ¥¹}½¬õQÉÕ”¤°(€€€€€€€€¤((€€€…ÁÀ¹µ½Õ¹Ð ˆ¼ˆ°MÑ…Ñ¥¥±•Ì¡‘¥É•Ñ½ÉäõÍÑ…Ñ¥}‘¥È°¡Ñµ°õQÉÕ”¤°¹…µ”ô‰ÍÑ…Ñ¥Œˆ¤((€€€É•ÑÕÉ¸…ÁÀ(()…ÁÀ€ôÉ•…Ñ•}…ÁÀ ¤(