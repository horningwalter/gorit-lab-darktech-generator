"""FastAPI worker that runs *inside* the Colab notebook on the GPU.

The Colab notebook does roughly::

    import os
    os.environ["DARKTECH_WORKER_API_KEY"] = "..."
    from darktech_generator.generation.remote_worker import build_app
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=8000)

A Cloudflare Tunnel running alongside (``cloudflared tunnel --url ...``)
publishes a public HTTPS URL that the local Gradio process points at via
``DARKTECH_REMOTE_URL``.

Cloudflare Tunnel kills HTTPS requests after ~100s, which is shorter than any
realistic ACE-Step generation. To work around that, the worker exposes an async
job protocol::

    POST /jobs       -> {job_id}                # returns immediately
    GET  /jobs/{id}  -> {status, progress, ...} # cheap, polled by the client

Generation runs in a background thread; the endpoint just reads job state.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import soundfile as sf

from darktech_generator.schemas import (
    RemoteGenerateRequest,
    RemoteGenerateResponse,
    RemoteJobAccepted,
    RemoteJobStatus,
)

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


class _JobStore:
    """Thread-safe in-memory job tracker. Survives only while the worker is up
    (which is fine: Colab sessions are ephemeral)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "status": "pending",
                "progress": 0.0,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "result": None,
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._jobs.get(job_id)
            return dict(data) if data is not None else None


def build_app() -> Any:
    """Construct the FastAPI app. Imported here so that fastapi stays optional locally."""
    from fastapi import FastAPI, Header, HTTPException

    from darktech_generator.config import get_settings
    from darktech_generator.generation.ace_step_renderer import AceStepRenderer

    settings = get_settings()
    settings.ensure_dirs()

    renderer = AceStepRenderer(cache_dir=settings.darktech_cache_dir)
    jobs = _JobStore()
    gen_lock = threading.Lock()  # serialize: ACE-Step does not handle parallel calls

    app = FastAPI(title="darktech-remote-worker", version="0.2.0")

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

    def _run_job(job_id: str, spec_payload: RemoteGenerateRequest) -> None:
        jobs.update(
            job_id,
            status="running",
            started_at=datetime.now(tz=timezone.utc),
        )
        started = time.perf_counter()
        try:
            with gen_lock:
                stem = renderer.render(spec_payload.spec)
            audio_b64 = _encode_wav_b64(stem.samples, stem.sample_rate)
            elapsed = time.perf_counter() - started
            result = RemoteGenerateResponse(
                audio_b64=audio_b64,
                sample_rate=stem.sample_rate,
                channels=int(stem.samples.shape[0]),
                actual_duration_s=stem.actual_duration_s,
                seed_used=stem.spec.seed or 0,
                elapsed_s=elapsed,
            )
            jobs.update(
                job_id,
                status="done",
                progress=1.0,
                finished_at=datetime.now(tz=timezone.utc),
                result=result,
            )
            logger.info("Job %s done in %.1fs", job_id, elapsed)
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Job %s failed", job_id)
            jobs.update(
                job_id,
                status="error",
                finished_at=datetime.now(tz=timezone.utc),
                error=f"{e}\n{tb}",
            )

    @app.post("/jobs", response_model=RemoteJobAccepted)
    def submit_job(
        request: RemoteGenerateRequest,
        x_api_key: str | None = Header(default=None),
    ) -> RemoteJobAccepted:
        _verify_api_key(x_api_key)
        job_id = jobs.create()
        thread = threading.Thread(target=_run_job, args=(job_id, request), daemon=True)
        thread.start()
        logger.info("Job %s submitted (stem=%s)", job_id, request.spec.name)
        return RemoteJobAccepted(job_id=job_id)

    @app.get("/jobs/{job_id}", response_model=RemoteJobStatus)
    def get_job(
        job_id: str,
        x_api_key: str | None = Header(default=None),
    ) -> RemoteJobStatus:
        _verify_api_key(x_api_key)
        data = jobs.get(job_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return RemoteJobStatus(job_id=job_id, **data)

    return app


def main() -> None:
    """Entry point if the worker is launched directly (rare; usually called from the notebook)."""
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
