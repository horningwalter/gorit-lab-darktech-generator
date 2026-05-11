"""Schema contract tests. These run offline, no GPU, no network."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from darktech_generator.schemas import (
    Bus,
    ReferenceProfile,
    RendererName,
    Section,
    StemSpec,
    TrackPlan,
)


def test_section_energy_bounds() -> None:
    Section(name="intro", bars=16, focus=[Bus.ATMOSPHERE], energy=0.3)
    with pytest.raises(ValidationError):
        Section(name="x", bars=16, energy=1.5)
    with pytest.raises(ValidationError):
        Section(name="x", bars=0)


def test_section_drops_unknown_focus_values() -> None:
    """DeepSeek occasionally invents bus names (vocal, drum). They should be
    silently dropped rather than rejecting the whole plan."""
    section = Section.model_validate(
        {"name": "drop_1", "bars": 32, "focus": ["kick", "vocal", "bass", "drum"]}
    )
    assert [b.value for b in section.focus] == ["kick", "bass"]


def test_stem_spec_rejects_darktech_literal() -> None:
    with pytest.raises(ValidationError):
        StemSpec(
            name="bad",
            bus=Bus.KICK,
            prompt="DarkTech kick at 190 bpm",
            duration_s=10,
            model=RendererName.ACE_STEP,
        )


def test_stem_spec_accepts_concrete_prompt() -> None:
    spec = StemSpec(
        name="kick_drop_1",
        bus=Bus.KICK,
        prompt="punchy distorted kick at 187 bpm, sub-heavy, dry",
        negative_prompt="vocals, melody",
        duration_s=30.0,
        seed=42,
        model=RendererName.ACE_STEP,
    )
    assert spec.seed == 42
    assert spec.cfg_scale == 6.0


def test_track_plan_unique_sections() -> None:
    with pytest.raises(ValidationError):
        TrackPlan(
            bpm=187,
            key="A minor",
            duration_seconds=180,
            sections=[
                Section(name="intro", bars=16),
                Section(name="intro", bars=16),
            ],
        )


def test_reference_profile_summary_empty() -> None:
    assert "No reference" in ReferenceProfile().to_llm_summary()


def test_reference_profile_summary_populated() -> None:
    rp = ReferenceProfile(
        source_paths=["/tmp/a.wav", "/tmp/b.wav"],
        bpm_median=187.5,
        key_dominant="A minor",
        lufs_integrated_median=-7.2,
        tags=["dark", "psychedelic", "tech"],
    )
    summary = rp.to_llm_summary()
    assert "187" in summary
    assert "A minor" in summary
    assert "dark" in summary
