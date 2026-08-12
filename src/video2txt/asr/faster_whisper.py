from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from video2txt.config import ASRSettings
from video2txt.media.probe import sha256_file
from video2txt.models import Transcript, TranscriptSegment, TranscriptWord


class ASRError(RuntimeError):
    """Raised when speech recognition cannot produce a usable result."""


def _log_probability_to_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, math.exp(value)))


class FasterWhisperEngine:
    def __init__(self, settings: ASRSettings, *, model: Any | None = None) -> None:
        self.settings = settings
        self._model = model

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.settings.model_path is None:
            raise ASRError(
                "未配置 faster-whisper 模型；请设置 --model-path、"
                "VIDEO2TXT_MODEL_PATH 或 config.toml"
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ASRError('未安装 ASR 依赖，请执行 pip install -e ".[asr]"') from error

        model_path = self.settings.model_path.expanduser()
        if not model_path.exists():
            raise ASRError(f"faster-whisper 模型目录不存在：{model_path}")
        self._model = WhisperModel(
            str(model_path.resolve()),
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            cpu_threads=self.settings.cpu_threads,
        )
        return self._model

    def transcribe(self, audio_path: Path) -> Transcript:
        audio = audio_path.resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        model = self._load_model()
        raw_segments, info = model.transcribe(
            str(audio),
            language=self.settings.language,
            task="transcribe",
            beam_size=self.settings.beam_size,
            word_timestamps=self.settings.word_timestamps,
            vad_filter=self.settings.vad_filter,
            condition_on_previous_text=self.settings.condition_on_previous_text,
        )

        segments: list[TranscriptSegment] = []
        for index, segment in enumerate(raw_segments, start=1):
            words = [
                TranscriptWord(
                    start=round(float(word.start), 3),
                    end=round(float(word.end), 3),
                    text=str(word.word).strip(),
                    probability=(
                        round(float(word.probability), 6)
                        if getattr(word, "probability", None) is not None
                        else None
                    ),
                )
                for word in (getattr(segment, "words", None) or [])
                if str(getattr(word, "word", "")).strip()
            ]
            text = str(segment.text).strip()
            if not text and not words:
                continue
            segments.append(
                TranscriptSegment(
                    id=f"asr-{index:04d}",
                    start=round(float(segment.start), 3),
                    end=round(float(segment.end), 3),
                    text=text,
                    confidence=_log_probability_to_confidence(
                        getattr(segment, "avg_logprob", None)
                    ),
                    words=words,
                )
            )

        if not segments:
            raise ASRError("音频中未识别到可用语音内容")

        model_name = str(self.settings.model_path or "injected-model")
        return Transcript(
            engine="faster-whisper",
            model=model_name,
            language=getattr(info, "language", self.settings.language),
            audio_sha256=sha256_file(audio),
            options={
                "language": self.settings.language,
                "task": "transcribe",
                "device": self.settings.device,
                "compute_type": self.settings.compute_type,
                "cpu_threads": self.settings.cpu_threads,
                "beam_size": self.settings.beam_size,
                "vad_filter": self.settings.vad_filter,
                "word_timestamps": self.settings.word_timestamps,
                "condition_on_previous_text": self.settings.condition_on_previous_text,
            },
            segments=segments,
        )
