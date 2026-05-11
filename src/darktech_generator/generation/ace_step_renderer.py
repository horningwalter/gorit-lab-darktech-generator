"""ACE-Step 1.5 renderer (primary stem generator).

ACE-Step is a diffusion text-to-music model. Standard checkpoint is ~3.5 GB and
runs in <4 GB VRAM on a Colab T4 free; the XL variant needs ~12 GB and is suited
to A100 with Colab Pro.

The official Python package (``ace-step``) exposes ``ACEStepPipeline`` with a
``generate`` method that accepts a prompt and produces a numpy array. Because the
package API is still moving (0.x), we keep all interaction confined to this file
so that future migrations are localized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from darktech_generator.generation.base import StemRenderer
from darktech_generator.schemas import AudioStem, StemSpec

logger = logging.getLogger(__name__)


class AceStepRenderer(StemRenderer):
    name = "ace_step"

    def __init__(
        self,
        cache_dir: Path,
        checkpoint: str = "ACE-Step/ACE-Step-v1-3.5B",
        device: str | None = None,
        dtype: str = "float16",
    ) -> None:
        self._cache_dir = cache_dir
        self._checkpoint = checkpoint
        self._device = device
        self._dtype_name = dtype
        self._pipeline: Any = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from acestep.pipeline_ace_step import ACEStepPipeline
        except ImportError as e:
            raise RuntimeError(
                "ACE-Step dependencies missing. Install with `pip install ace-step torch`."
            ) from e

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            self._dtype_name
        ]
        logger.info(
            "Loading ACE-Step checkpoint=%s on device=%s dtype=%s",
            self._checkpoint,
            device,
            self._dtype_name,
        )
        self._pipeline = ACEStepPipeline(
            checkpoint_dir=str(self._cache_dir / "ace_step"),
            dtype=dtype,
            device=device,
            torch_compile=False,
        )

    def render(self, spec: StemSpec) -> AudioStem:
        self._ensure_loaded()
        assert self._pipeline is not None

        seed = spec.seed if spec.seed is not None else int(np.random.randint(0, 2**31 - 1))

        logger.info(
            "ACE-Step rendering stem=%s duration=%.1fs seed=%d",
            spec.name,
            spec.duration_s,
            seed,
        )
        result = self._pipeline(
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt or None,
            audio_duration=spec.duration_s,
            guidance_scale=spec.cfg_scale,
            num_inference_steps=spec.steps,
            manual_seeds=str(seed),
        )

        samples, sample_rate = self._extract_audio(result)
        samples = self._to_float32_stereo(samples)
        actual_duration = samples.shape[-1] / float(sample_rate)

        return AudioStem(
            spec=spec.model_copy(update={"seed": seed}),
            samples=samples,
            sample_rate=sample_rate,
            actual_duration_s=actual_duration,
        )

    @staticmethod
    def _extract_audio(result: Any) -> tuple[np.ndarray, int]:
        """ACE-Step's return shape has shifted across versions; normalize it here."""
        if isinstance(result, tuple) and len(result) == 2:
            samples, sample_rate = result
        elif isinstance(result, dict):
            samples = result["audio"]
            sample_rate = int(result.get("sample_rate", 44100))
        else:
            samples = result
            sample_rate = 44100
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()
        return np.asarray(samples), int(sample_rate)

    @staticmethod
    def _to_float32_stereo(samples: np.ndarray) -> np.ndarray:
        arr = samples.astype(np.float32, copy=False)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=0)
        elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = np.concatenate([arr, arr], axis=0)
        peak = float(np.max(np.abs(arr))) if arr.size else 1.0
        if peak > 1.0:
            arr = arr / peak
        return arr

    def unload(self) -> None:
        if self._pipeline is None:
            return
        try:
            import torch

            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("Failed to fully unload ACE-Step: %s", e)
