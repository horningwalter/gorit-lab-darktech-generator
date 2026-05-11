"""FastAPI worker that runs *inside* the Colab notebook on the A100.

The Colab notebook does roughly::

    import os
    os.environ["DARKTECH_WORKER_API_KEY"] = "..."
    from darktech_generator.generation.remote_worker import build_app
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=8000)

A Cloudflare Tunnel running alongside (``cloudflared tunnel --url http://localhost:8000``)
publishes a stable HTTPS URL that the local Gradio process points at via
``DARKTECH_REMOTE_URL``.

This module imports ACE-Step lazily so that ``import darktech_generator`` stays
cheap on machines without torch.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any

import numpy as np
import soundfile as sf

from darktech_generator.schemas import RemoteGenerateRequest, RemoteGenerateResponse

logger = logging.getLogger(__name__)

_WORKER_API_KEY_ENV = "DARKTECH_WORKER_API_KEY"


def _verify_api_key(provided: str | None) -> None:
    expected = os.environ.get(_WORKER_API_KEY_ENV, "")
    if not expected:
        return
    if provided != expected:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


def _encode_wav_b64(samples: np.ndarray, sample_rate: int) -> str:
    if samples.ndim != 2:
        raise ValueError(f"Expected (channels, samples), got {samples.shape}")
    buf = io.BytesIO()
    sf.write(buf, samples.T, sample_rate, format="WAV", subtype="PCM_24")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_app() -> Any:
    """Construct the FastAPI app. Imported here so that fastapi stays optional locally."""
    from fastapi import FastAPI, Header, HTTPException

    from darktech_generator.generation.ace_step_renderer import AceStepRenderer
    from darktech_generator.config import get_settings

    settings = get_settings()
    settings.ensure_dirs()

    renderer = AceStepRenderer(cache_dir=settings.darktech_cache_dir)

    app = FastAPI(title="darktech-remote-worker", version="0.1.0")

    @app.get("/health")
    def health(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _verify_api_key(x_api_key)
        gpu_info: dict[str, Any] = {"cuda_available": False}
        try:
            import torch

            gpu_info["cuda_available"] = torch.cuda.is_available()
            if gpu_info["cuda_available"]:
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_info["vram_gb"] = round(props.total_memory / (1024**3), 1)
                gpu_info["vram_free_gb"] = round(
                    torch.cuda.mem_get_info()[0] / (1024**3), 1
                )
        except ImportError:
            pass
        return {"status": "ok", "renderer": "ace_step", "gpu": gpu_info}

    @app.post("/generate", response_model=RemoteGenerateResponse)
    def generate(
        request: RemoteGenerateRequest,
        x_api_key: str | None = Header(default=None),
    ) -> RemoteGenerateResponse:
        _verify_api_key(x_api_key)
        started = time.perf_counter()
        try:
            stem = renderer.render(request.spec)
        except Exception as e:
            logger.exception("Generation failed for stem=%s", request.spec.name)
            raise HTTPException(status_code=500, detail=str(e)) from e

        audio_b64 = _encode_wav_b64(stem.samples, stem.sample_rate)
        elapsed = time.perf_counter() - started
        return RemoteGenerateResponse(
            audio_b64=audio_b64,
            sample_rate=stem.sample_rate,
            channels=int(stem.samples.shape[0]),
            actual_duration_s=stem.actual_duration_s,
            seed_used=stem.spec.seed or 0,
            elapsed_s=elapsed,
        )

    return app


def main() -> None:
    """Entry point if the worker is launched directly (rare; usually called from the notebook)."""
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
