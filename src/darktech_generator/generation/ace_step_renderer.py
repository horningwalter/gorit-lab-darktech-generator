"""ACE-Step 1.5 renderer (primary stem generator).

ACE-Step is a diffusion text-to-music model. Standard checkpoint is ~3.5 GB and
runs in <4 GB VRAM on a Colab T4; the XL variant needs ~12 GB and is suited to
A100 with Colab Pro.

Pipeline API reference (ACE-Step 0.x):
    ACEStepPipeline(checkpoint_dir, device_id=0, dtype="bfloat16", ...)
    pipeline(
        prompt: str,
        lyrics: str | None,           # we leave None: DarkTech is instrumental
        audio_duration: float,
        infer_step: int,
        guidance_scale: float,
        manual_seeds: list[int],
        save_path: str,               # ACE-Step writes WAVs to disk
        format: str = "wav",
        ...
    )
    -> list[str] + [input_params_json_dict]

The pipeline writes WAVs to ``save_path`` and returns the list of paths. We read
the WAV back from disk to honor the in-memory :class:`AudioStem` contract.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from darktech_generator.generation.base import StemRenderer
from darktech_generator.schemas import AudioStem, StemSpec

logger = logging.getLogger(__name__)


class AceStepRenderer(StemRenderer):
    name = "ace_step"

    def __init__(
        self,
        cache_dir: Path,
        device_id: int = 0,
        dtype: str = "bfloat16",
    ) -> None:
        self._cache_dir = cache_dir
        self._device_id = device_id
        self._dtype_name = dtype
        self._pipeline: Any = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "torch not installed. Install the [ml] extra: pip install -e '.[ml]'."
            ) from e

        ACEStepPipeline = self._import_pipeline()
        checkpoint_dir = str(self._cache_dir / "ace_step")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(
            "Loading ACE-Step checkpoint_dir=%s device_id=%d dtype=%s",
            checkpoint_dir,
            self._device_id,
            self._dtype_name,
        )
        self._pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_dir,
            device_id=self._device_id,
            dtype=self._dtype_name,
            torch_compile=False,
        )

    def render(self, spec: StemSpec) -> AudioStem:
        self._ensure_loaded()
        assert self._pipeline is not None

        seed = spec.seed if spec.seed is not None else int(np.random.randint(0, 2**31 - 1))

        # ACE-Step writes WAVs to disk. Give it a private temp dir per call so
        # we don't pile artifacts.
        with tempfile.TemporaryDirectory(prefix="acestep_render_") as tmp_dir:
            output_dir = Path(tmp_dir)
            logger.info(
                "ACE-Step rendering stem=%s duration=%.1fs seed=%d -> %s",
                spec.name,
                spec.duration_s,
                seed,
                output_dir,
            )
            call_kwargs = {
                "prompt": spec.prompt,
                "lyrics": None,
                "audio_duration": float(spec.duration_s),
                "infer_step": int(spec.steps),
                "guidance_scale": float(spec.cfg_scale),
                "manual_seeds": [int(seed)],
                "save_path": str(output_dir),
                "format": "wav",
            }
            if spec.reference_audio_path:
                call_kwargs.update(
                    audio2audio_enable=True,
                    ref_audio_input=spec.reference_audio_path,
                )
            result = self._pipeline(**call_kwargs)

        wav_paths = [p for p in self._flatten(result) if isinstance(p, str) and p.endswith(".wav")]
        if not wav_paths:
            raise RuntimeError(
                f"ACE-Step returned no WAV paths; got {type(result).__name__}: {result!r}"
            )

        samples, sample_rate = self._read_wav(wav_paths[0])
        samples = self._to_float32_stereo(samples)
        actual_duration = samples.shape[-1] / float(sample_rate)

        return AudioStem(
            spec=spec.model_copy(update={"seed": seed}),
            samples=samples,
            sample_rate=sample_rate,
            actual_duration_s=actual_duration,
        )

    @staticmethod
    def _import_pipeline() -> Any:
        """Try several known import paths for ACE-Step.

        The package layout has changed across releases. We try the most likely
        modules in order and raise an informative error if none work.
        """
        candidates = [
            ("acestep.pipeline_ace_step", "ACEStepPipeline"),
            ("acestep.pipeline", "ACEStepPipeline"),
            ("ace_step.pipeline_ace_step", "ACEStepPipeline"),
            ("ace_step.pipeline", "ACEStepPipeline"),
        ]
        last_error: ImportError | None = None
        for module_name, attr in candidates:
            try:
                module = __import__(module_name, fromlist=[attr])
                return getattr(module, attr)
            except ImportError as e:
                last_error = e
                continue
            except AttributeError as e:
                last_error = ImportError(str(e))
                continue
        raise RuntimeError(
            "ACE-Step not installed or import path unknown. In Colab, run:\n"
            "  !pip install -q git+https://github.com/ace-step/ACE-Step.git\n"
            f"Last error: {last_error}"
        )

    @staticmethod
    def _flatten(obj: Any) -> list[Any]:
        """ACE-Step returns ``output_paths + [params_dict]``. Flatten nested lists."""
        out: list[Any] = []
        if isinstance(obj, (list, tuple)):
            for item in obj:
                out.extend(AceStepRenderer._flatten(item))
        else:
            out.append(obj)
        return out

    @staticmethod
    def _read_wav(path: str) -> tuple[np.ndarray, int]:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return np.asarray(data, dtype=np.float32).T, int(sr)

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
