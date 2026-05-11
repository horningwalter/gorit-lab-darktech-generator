"""Public Python API: programmatic one-shot generation + Gradio launcher.

The MVP path is deliberately linear:
    user_prompt -> OrchestratorAgent.plan -> single ACE-Step render -> master -> export.

Multi-stem generation, reference analysis, and the iteration loop arrive in Phase 2+.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from darktech_generator.audio.exporter import export_metadata, export_wav
from darktech_generator.audio.mastering import master
from darktech_generator.config import get_settings
from darktech_generator.generation.model_registry import ModelRegistry
from darktech_generator.orchestration.agent import OrchestratorAgent
from darktech_generator.schemas import (
    Bus,
    ExtensionStrategy,
    GenerationResult,
    RendererName,
    StemSpec,
    TrackPlan,
)

logger = logging.getLogger(__name__)


def _mvp_stem_spec_from_plan(plan: TrackPlan) -> StemSpec:
    """For the MVP we generate one full-mix stem of up to 60s from the plan notes.

    Phase 2 replaces this with per-bus StemSpecs from ``plan_stems``.
    """
    duration = min(plan.duration_seconds, 60.0)
    prompt = (
        f"aggressive distorted kick at {plan.bpm:.0f} bpm, sub-heavy rolling bass with "
        f"FM saturation, dark psychedelic atmospheric pad, polyrhythmic tech percussion, "
        f"key of {plan.key.lower()}, mechanical and dry mix"
    )
    return StemSpec(
        name="mvp_full_mix",
        bus=Bus.KICK,
        prompt=prompt,
        negative_prompt="vocals, melody, soft, ambient, major key, happy, slow",
        duration_s=duration,
        cfg_scale=7.0,
        steps=30,
        model=RendererName.ACE_STEP,
        extension_strategy=ExtensionStrategy.GENERATE_LONG,
    )


async def _generate_async(
    user_prompt: str,
    output_dir: Path | None = None,
) -> GenerationResult:
    settings = get_settings()
    settings.ensure_dirs()
    output_root = output_dir or settings.darktech_output_dir

    async with OrchestratorAgent() as agent:
        plan = await agent.plan(user_prompt)
        cost_entries = list(agent.client.cost_tracker.entries)

    logger.info("TrackPlan ready: BPM=%.1f, key=%s, sections=%d", plan.bpm, plan.key, len(plan.sections))

    spec = _mvp_stem_spec_from_plan(plan)
    renderer = ModelRegistry.instance().get(RendererName.ACE_STEP)
    stem = renderer.render(spec)

    mastered = master(stem.samples, stem.sample_rate)

    session_started = datetime.now(tz=timezone.utc)
    session_dir = output_root / f"session_{session_started.strftime('%Y%m%d_%H%M%S')}"
    wav_path = export_wav(
        mastered,
        stem.sample_rate,
        session_dir / "track.wav",
        bit_depth=settings.output_bit_depth,
    )

    result = GenerationResult(
        started_at=session_started,
        finished_at=datetime.now(tz=timezone.utc),
        user_prompt=user_prompt,
        track_plan=plan,
        output_wav_path=str(wav_path),
        costs=cost_entries,
        seeds_used={spec.name: stem.spec.seed} if stem.spec.seed is not None else {},
    )
    export_metadata(result, session_dir / "metadata.json")
    return result


def generate_track(user_prompt: str, output_dir: Path | None = None) -> GenerationResult:
    """Synchronous wrapper around the async pipeline; convenient for notebooks."""
    return asyncio.run(_generate_async(user_prompt, output_dir=output_dir))


def launch_gradio(share: bool | None = None) -> None:
    """Launch the Gradio UI. Imported lazily to keep import time low for the API path."""
    from darktech_generator.ui.gradio_app import launch

    launch(share=share)
