"""RemoteRenderer round-trip with httpx MockTransport. Offline, no real network."""

from __future__ import annotations

import base64
import io
import os

import httpx
import numpy as np
import pytest
import soundfile as sf

from darktech_generator.config import reset_settings_for_tests
from darktech_generator.generation.remote_renderer import RemoteRenderer
from darktech_generator.schemas import (
    Bus,
    RemoteGenerateRequest,
    RemoteGenerateResponse,
    RendererName,
    StemSpec,
)


def _make_wav_b64(sample_rate: int = 44100, seconds: float = 0.5) -> str:
    n = int(sample_rate * seconds)
    samples = (np.random.default_rng(0).standard_normal((2, n)) * 0.05).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples.T, sample_rate, format="WAV", subtype="PCM_24")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_remote_renderer_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DARKTECH_REMOTE_URL", "https://worker.example")
    monkeypatch.setenv("DARKTECH_REMOTE_API_KEY", "test-key")
    reset_settings_for_tests()

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("X-API-Key")
        body = RemoteGenerateRequest.model_validate_json(request.content)
        captured["spec_name"] = body.spec.name
        wav_b64 = _make_wav_b64()
        return httpx.Response(
            200,
            json=RemoteGenerateResponse(
                audio_b64=wav_b64,
                sample_rate=44100,
                channels=2,
                actual_duration_s=0.5,
                seed_used=123,
                elapsed_s=0.42,
            ).model_dump(),
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    class PatchedClient(original_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("darktech_generator.generation.remote_renderer.httpx.Client", PatchedClient)

    renderer = RemoteRenderer()
    spec = StemSpec(
        name="kick_drop_1",
        bus=Bus.KICK,
        prompt="punchy distorted kick at 187 bpm, sub-heavy, dry",
        duration_s=30.0,
        model=RendererName.ACE_STEP,
    )
    stem = renderer.render(spec)

    assert captured["url"] == "https://worker.example/generate"
    assert captured["api_key"] == "test-key"
    assert captured["spec_name"] == "kick_drop_1"
    assert stem.sample_rate == 44100
    assert stem.samples.shape[0] == 2
    assert stem.samples.dtype == np.float32
    assert stem.spec.seed == 123


def test_remote_renderer_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DARKTECH_REMOTE_URL", "")
    reset_settings_for_tests()
    try:
        with pytest.raises(RuntimeError, match="DARKTECH_REMOTE_URL"):
            RemoteRenderer()
    finally:
        os.environ.pop("DARKTECH_REMOTE_URL", None)
        reset_settings_for_tests()
