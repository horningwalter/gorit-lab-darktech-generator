"""Core data contracts shared across orchestration, generation, and audio layers.

All inter-layer communication uses these Pydantic models. The orchestration layer
produces them from the LLM (DeepSeek V4 Pro) and the generation/audio layers consume
them. Keeping this as the single source of truth lets us validate LLM output
end-to-end and catch malformed JSON at the boundary instead of mid-pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Bus(str, Enum):
    KICK = "kick"
    BASS = "bass"
    PERCUSSION = "percussion"
    ATMOSPHERE = "atmosphere"
    TECH = "tech"
    LEAD = "lead"
    FX = "fx"


class ExtensionStrategy(str, Enum):
    LOOP = "loop"
    CROSSFADE = "crossfade"
    GENERATE_LONG = "generate_long"
    ONE_SHOT = "one_shot"


class RendererName(str, Enum):
    ACE_STEP = "ace_step"
    STABLE_AUDIO = "stable_audio"
    MUSICGEN_MELODY = "musicgen_melody"


class Section(BaseModel):
    name: str = Field(description="Section label, e.g. intro, buildup, drop_1, break, outro.")
    bars: int = Field(ge=1, le=256, description="Length in bars at the track BPM.")
    focus: list[Bus] = Field(
        default_factory=list,
        description="Which buses dominate this section. Used to gate stem generation.",
    )
    energy: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0=ambient, 1=peak. Influences mix automation and stem prompts.",
    )

    @field_validator("focus", mode="before")
    @classmethod
    def _drop_unknown_focus(cls, v: object) -> object:
        """LLMs occasionally hallucinate buses (vocal, drum, etc). Drop them silently
        instead of rejecting the whole plan; the supported buses cover all DarkTech
        material."""
        if not isinstance(v, list):
            return v
        allowed = {b.value for b in Bus}
        kept: list[object] = []
        for item in v:
            if isinstance(item, Bus):
                kept.append(item)
            elif isinstance(item, str) and item in allowed:
                kept.append(item)
        return kept


class StemSpec(BaseModel):
    name: str = Field(description="Unique identifier within a TrackPlan (e.g. kick_bus_drop_1).")
    bus: Bus
    prompt: str = Field(
        description=(
            "Text-to-audio prompt for the renderer. Must be concrete and timbral, "
            "never use 'DarkTech' literally; describe the sound."
        )
    )
    negative_prompt: str = Field(
        default="",
        description="Comma-separated terms to push the renderer away from (e.g. 'vocals, melody').",
    )
    duration_s: float = Field(ge=1.0, le=240.0, description="Target output duration in seconds.")
    seed: int | None = Field(default=None, ge=0)
    cfg_scale: float = Field(default=6.0, ge=1.0, le=20.0)
    steps: int = Field(default=30, ge=1, le=200)
    model: RendererName = RendererName.ACE_STEP
    extension_strategy: ExtensionStrategy = ExtensionStrategy.LOOP
    reference_audio_path: str | None = Field(
        default=None,
        description="Optional path to a reference WAV for audio2audio / melody conditioning.",
    )

    @field_validator("prompt")
    @classmethod
    def _no_genre_literal(cls, v: str) -> str:
        if "darktech" in v.lower():
            raise ValueError(
                "StemSpec.prompt must not contain the literal 'DarkTech'; describe the sound."
            )
        return v


class TrackPlan(BaseModel):
    """High-level plan emitted by DeepSeek V4 Pro from the user prompt + references."""

    bpm: float = Field(ge=60.0, le=240.0)
    key: str = Field(description="Tonality, e.g. 'A minor', 'F# minor'.")
    duration_seconds: float = Field(ge=30.0, le=720.0)
    sections: list[Section] = Field(min_length=1)
    stem_specs: list[StemSpec] = Field(default_factory=list)
    notes: str = Field(default="", description="Director-level notes about intent and references.")

    @field_validator("sections")
    @classmethod
    def _sections_unique(cls, v: list[Section]) -> list[Section]:
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Section names must be unique within a TrackPlan.")
        return v


class ReferenceProfile(BaseModel):
    """Aggregated analysis of one or more reference tracks supplied by the user."""

    source_paths: list[str] = Field(default_factory=list)
    bpm_median: float | None = None
    key_dominant: str | None = None
    lufs_integrated_median: float | None = None
    tags: list[str] = Field(default_factory=list)
    mert_embedding_mean: list[float] | None = Field(
        default=None,
        description="Mean MERT embedding across references, 1024-dim (or None if not computed).",
    )
    clap_embedding_mean: list[float] | None = Field(
        default=None,
        description="Mean LAION-CLAP embedding across references, 512-dim (or None).",
    )

    def to_llm_summary(self) -> str:
        """Compact textual form for injection into the DeepSeek system prompt."""
        if not self.source_paths:
            return "No reference tracks provided."
        parts = [f"References: {len(self.source_paths)} track(s)."]
        if self.bpm_median is not None:
            parts.append(f"BPM median {self.bpm_median:.1f}.")
        if self.key_dominant is not None:
            parts.append(f"Key {self.key_dominant}.")
        if self.lufs_integrated_median is not None:
            parts.append(f"LUFS-i median {self.lufs_integrated_median:.1f}.")
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags[:10]) + ".")
        return " ".join(parts)


class AudioStem(BaseModel):
    """A rendered stem in memory. Samples are not serialized; only metadata is."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: StemSpec
    samples: np.ndarray = Field(description="Shape (channels, num_samples), float32 in [-1, 1].")
    sample_rate: int = Field(ge=8000, le=192000)
    actual_duration_s: float = Field(ge=0.0)


class CostEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    provider: Literal["deepseek", "huggingface", "colab"] = "deepseek"
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class GenerationResult(BaseModel):
    """Final artifact of one end-to-end run; serialized as metadata.json alongside WAVs."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    finished_at: datetime | None = None
    user_prompt: str
    reference_profile: ReferenceProfile | None = None
    track_plan: TrackPlan
    output_wav_path: str
    stem_wav_paths: dict[str, str] = Field(default_factory=dict)
    costs: list[CostEntry] = Field(default_factory=list)
    seeds_used: dict[str, int] = Field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return sum(c.usd for c in self.costs)


class RemoteGenerateRequest(BaseModel):
    """Payload sent from the local renderer to the Colab worker over HTTPS."""

    spec: StemSpec


class RemoteGenerateResponse(BaseModel):
    """Worker response. ``audio_b64`` is a base64-encoded WAV (raw bytes)."""

    audio_b64: str
    sample_rate: int = Field(ge=8000, le=192000)
    channels: int = Field(ge=1, le=2)
    actual_duration_s: float = Field(ge=0.0)
    seed_used: int
    elapsed_s: float = Field(ge=0.0)


class RevisionInstructions(BaseModel):
    """Output of the critique loop: which stems to regenerate and how to nudge the prompts."""

    target_stems: list[str] = Field(min_length=1, description="StemSpec.name values to regenerate.")
    prompt_patches: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of StemSpec.name -> additional prompt fragment to append.",
    )
    seed_strategy: Literal["keep", "increment", "random"] = "increment"
    notes: str = ""
