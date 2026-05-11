"""Lazy registry of audio model renderers, cached across calls.

Models are downloaded once into ``Settings.darktech_cache_dir`` (which is symlinked
to Google Drive when running in Colab) and kept in memory for the lifetime of the
session.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from darktech_generator.config import get_settings
from darktech_generator.generation.base import StemRenderer
from darktech_generator.schemas import RendererName


class ModelRegistry:
    _instance: "ModelRegistry | None" = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._renderers: dict[RendererName, StemRenderer] = {}
        self._lock = Lock()
        self._settings = get_settings()

    @classmethod
    def instance(cls) -> "ModelRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get(self, name: RendererName, **kwargs: Any) -> StemRenderer:
        with self._lock:
            if name not in self._renderers:
                self._renderers[name] = self._build(name, **kwargs)
            return self._renderers[name]

    def _build(self, name: RendererName, **kwargs: Any) -> StemRenderer:
        if name == RendererName.ACE_STEP:
            if self._settings.darktech_remote_url:
                from darktech_generator.generation.remote_renderer import RemoteRenderer

                return RemoteRenderer()
            from darktech_generator.generation.ace_step_renderer import AceStepRenderer

            return AceStepRenderer(
                cache_dir=self._settings.darktech_cache_dir,
                **kwargs,
            )
        if name == RendererName.STABLE_AUDIO:
            from darktech_generator.generation.stable_audio_renderer import (
                StableAudioRenderer,
            )

            return StableAudioRenderer(
                cache_dir=self._settings.darktech_cache_dir,
                **kwargs,
            )
        if name == RendererName.MUSICGEN_MELODY:
            from darktech_generator.generation.musicgen_melody_renderer import (
                MusicgenMelodyRenderer,
            )

            return MusicgenMelodyRenderer(
                cache_dir=self._settings.darktech_cache_dir,
                **kwargs,
            )
        raise ValueError(f"Unknown renderer: {name}")

    def unload_all(self) -> None:
        with self._lock:
            for renderer in self._renderers.values():
                try:
                    renderer.unload()
                except Exception:
                    pass
            self._renderers.clear()
