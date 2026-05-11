"""Renderer protocol shared by all generation backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from darktech_generator.schemas import AudioStem, StemSpec


@runtime_checkable
class StemRenderer(Protocol):
    """A backend that converts a :class:`StemSpec` into an :class:`AudioStem`."""

    name: str

    def render(self, spec: StemSpec) -> AudioStem:
        """Produce audio for the given spec. May download model weights on first call."""
        ...

    def unload(self) -> None:
        """Free GPU memory held by the model."""
        ...
