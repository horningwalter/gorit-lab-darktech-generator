"""Mastering bus: limiter + LUFS normalization. Phase 4 will add multiband comp."""

from __future__ import annotations

import logging

import numpy as np

from darktech_generator.config import get_settings

logger = logging.getLogger(__name__)


def master(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply a minimal mastering chain and return float32 audio.

    Chain: pedalboard limiter (true-peak target from settings) -> pyloudnorm LUFS
    normalization (integrated target from settings).

    Shape conventions:
        Input: (channels, num_samples), float32 in [-1, 1].
        Output: same shape, peak-limited and loudness-normalized.
    """
    if samples.ndim != 2:
        raise ValueError(f"Expected (channels, samples) ndarray, got shape {samples.shape}.")
    if samples.shape[0] not in (1, 2):
        raise ValueError(f"Expected mono or stereo audio, got {samples.shape[0]} channels.")

    settings = get_settings()
    limited = _apply_limiter(samples, sample_rate, settings.target_true_peak_db)
    normalized = _apply_lufs(limited, sample_rate, settings.target_lufs_integrated)
    return _final_safety_clip(normalized)


def _apply_limiter(samples: np.ndarray, sample_rate: int, threshold_db: float) -> np.ndarray:
    try:
        from pedalboard import Limiter, Pedalboard
    except ImportError as e:
        raise RuntimeError("pedalboard is required for mastering. Run `pip install pedalboard`.") from e

    board = Pedalboard([Limiter(threshold_db=threshold_db, release_ms=80.0)])
    pb_input = samples.T if samples.shape[0] in (1, 2) else samples
    out = board(pb_input, sample_rate=sample_rate)
    return np.asarray(out, dtype=np.float32).T


def _apply_lufs(samples: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    try:
        import pyloudnorm as pyln
    except ImportError as e:
        raise RuntimeError("pyloudnorm is required for mastering.") from e

    meter = pyln.Meter(sample_rate)
    measure_input = samples.T if samples.shape[0] in (1, 2) else samples
    loudness = meter.integrated_loudness(measure_input)
    if not np.isfinite(loudness):
        logger.warning("LUFS measurement returned %s; skipping LUFS normalization.", loudness)
        return samples
    normalized = pyln.normalize.loudness(measure_input, loudness, target_lufs)
    return np.asarray(normalized, dtype=np.float32).T


def _final_safety_clip(samples: np.ndarray, ceiling: float = 0.999) -> np.ndarray:
    return np.clip(samples, -ceiling, ceiling).astype(np.float32, copy=False)
