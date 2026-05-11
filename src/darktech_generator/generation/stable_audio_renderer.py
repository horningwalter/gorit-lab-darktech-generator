"""Stable Audio Open 1.0 renderer (fallback for ACE-Step).

Not implemented in the MVP. Stubbed to satisfy the model registry contract; full
implementation lands in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

from darktech_generator.generation.base import StemRenderer
from darktech_generator.schemas import AudioStem, StemSpec


class StableAudioRenderer(StemRenderer):
    name = "stable_audio"

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def render(self, spec: StemSpec) -> AudioStem:
        raise NotImplementedError(
            "StableAudioRenderer lands in Phase 2. Use RendererName.ACE_STEP in the MVP."
        )

    def unload(self) -> None:
        return
