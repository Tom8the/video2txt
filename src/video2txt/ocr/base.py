from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video2txt.models import OCRLine


class OCREngine(Protocol):
    def recognize(self, image_path: Path) -> list[OCRLine]: ...
