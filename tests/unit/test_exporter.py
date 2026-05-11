"""Exporter writes valid WAVs and JSON sidecars."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from darktech_generator.audio.exporter import export_metadata, export_wav
from darktech_generator.schemas import (
    Bus,
    GenerationResult,
    RendererName,
    Section,
    StemSpec,
    TrackPlan,
)


def test_export_wav_roundtrip(tmp_path: Path) -> None:
    sr = 44100
    samples = (np.random.default_rng(0).standard_normal((2, sr)) * 0.1).astype(np.float32)
    path = export_wav(samples, sr, tmp_path / "out.wav", bit_depth=24)
    assert path.exists()
    data, read_sr = sf.read(str(path))
    assert read_sr == sr
    assert data.shape == (sr, 2)


def test_export_metadata_writes_valid_json(tmp_path: Path) -> None:
    plan = TrackPlan(
        bpm=187,
        key="A minor",
        duration_seconds=60,
        sections=[Section(name="intro", bars=16, focus=[Bus.ATMOSPHERE])],
        stem_specs=[
            StemSpec(
                name="kick",
                bus=Bus.KICK,
                prompt="punchy distorted kick at 187 bpm",
                duration_s=30,
                model=RendererName.ACE_STEP,
            )
        ],
    )
    result = GenerationResult(
        user_prompt="test",
        track_plan=plan,
        output_wav_path=str(tmp_path / "track.wav"),
    )
    path = export_metadata(result, tmp_path / "meta.json")
    import json

    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["user_prompt"] == "test"
    assert parsed["track_plan"]["bpm"] == 187


def test_export_wav_rejects_bad_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_wav(np.zeros(1000, dtype=np.float32), 44100, tmp_path / "x.wav")
