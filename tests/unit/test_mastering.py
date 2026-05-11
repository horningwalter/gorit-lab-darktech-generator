"""Mastering chain smoke test. Synthesizes audio in CPU, no GPU/network."""

from __future__ import annotations

import numpy as np
import pytest

from darktech_generator.audio.mastering import master


@pytest.fixture
def stereo_sine() -> tuple[np.ndarray, int]:
    sr = 44100
    duration_s = 2.0
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    left = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.3 * np.sin(2 * np.pi * 330.0 * t)
    samples = np.stack([left, right], axis=0)
    return samples, sr


def test_master_output_shape_and_peak(stereo_sine: tuple[np.ndarray, int]) -> None:
    pytest.importorskip("pedalboard")
    pytest.importorskip("pyloudnorm")

    samples, sr = stereo_sine
    out = master(samples, sr)

    assert out.shape == samples.shape
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) < 1.0


def test_master_rejects_wrong_shape() -> None:
    pytest.importorskip("pedalboard")
    pytest.importorskip("pyloudnorm")

    with pytest.raises(ValueError):
        master(np.zeros(1000, dtype=np.float32), 44100)
    with pytest.raises(ValueError):
        master(np.zeros((5, 1000), dtype=np.float32), 44100)
