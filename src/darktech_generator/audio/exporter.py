"""WAV export with bit-depth control and JSON metadata sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from darktech_generator.schemas import GenerationResult


def export_wav(samples: np.ndarray, sample_rate: int, path: Path, bit_depth: int = 24) -> Path:
    """Write a (channels, samples) float32 array to a WAV file at the given bit depth."""
    if samples.ndim != 2:
        raise ValueError(f"Expected (channels, samples), got shape {samples.shape}.")
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth)
    if subtype is None:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.T, sample_rate, subtype=subtype)
    return path


def export_metadata(result: GenerationResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
