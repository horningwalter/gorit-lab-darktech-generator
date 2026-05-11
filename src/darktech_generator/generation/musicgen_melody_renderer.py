"""MusicGen-Melody renderer (audio-conditioned generation).

Not implemented in the MVP. Phase 2 wires this in for the reference-conditioning
path (when the user uploads a track to imitate stylistically).
"""

from __future__ import annotations

from pathlib import Path

from darktech_generator.generation.base import StemRenderer
from darktech_generator.schemas import AudioStem, StemSpec


class MusicgenMelodyRenderer(StemRenderer):
    name = "musicgen_melody"

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def render(self, spec: StemSpec) -> AudioStem:
        raise NotImplementedError(
            "MusicgenMelodyRenderer lands in Phase 2 for reference-conditioned stems."
        )

    def unload(self) -> None:
        return
