from pathlib import Path

import pytest
from pydantic import ValidationError

from video2txt.config import AlignmentSettings, load_settings


def test_default_settings_match_verified_asr_baseline() -> None:
    settings = load_settings()

    assert settings.asr.language == "zh"
    assert settings.asr.device == "cpu"
    assert settings.asr.compute_type == "int8"
    assert settings.asr.cpu_threads == 4
    assert settings.asr.beam_size == 5
    assert settings.asr.vad_filter is True
    assert settings.asr.word_timestamps is True
    assert settings.asr.condition_on_previous_text is False


def test_load_settings_applies_model_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[asr]\nmodel_path = "from-config"\n', encoding="utf-8")
    monkeypatch.setenv("VIDEO2TXT_MODEL_PATH", str(tmp_path / "from-env"))

    settings = load_settings(config)

    assert settings.asr.model_path == tmp_path / "from-env"


def test_alignment_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        AlignmentSettings(time_weight=0.5, text_weight=0.5, confidence_weight=0.5)

