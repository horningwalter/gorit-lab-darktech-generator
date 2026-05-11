"""HTTP client that delegates rendering to a remote Colab worker.

Uses an async job protocol because Cloudflare Tunnel kills HTTPS requests after
~100 seconds, but audio generation takes minutes. Flow:

    POST /jobs        -> {job_id}
    GET  /jobs/{id}   -> {status, progress, result?}    (poll every ~5s)

Each individual request finishes in 1-3s, well under the proxy timeout. The
heavy work runs in a background thread on the worker side.
"""

from __future__ import annotations

import base64
import io
import logging
import time

import httpx
import numpy as np
import soundfile as sf
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from darktech_generator.config import get_settings
from darktech_generator.generation.base import StemRenderer
from darktech_generator.schemas import (
    AudioStem,
    RemoteGenerateRequest,
    RemoteJobAccepted,
    RemoteJobStatus,
    StemSpec,
)

logger = logging.getLogger(__name__)


class RemoteRenderer(StemRenderer):
    name = "remote"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float = 5.0,
        max_wait_s: float = 1800.0,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.darktech_remote_url).rstrip("/")
        self._api_key = api_key or settings.darktech_remote_api_key.get_secret_value()
        self._timeout_s = timeout_s or settings.darktech_remote_timeout_s
        self._poll_interval_s = poll_interval_s
        self._max_wait_s = max_wait_s
        if not self._base_url:
            raise RuntimeError(
                "RemoteRenderer requires DARKTECH_REMOTE_URL to be set."
            )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    def render(self, spec: StemSpec) -> AudioStem:
        job_id = self._submit_job(spec)
        logger.info("RemoteRenderer submitted job=%s for stem=%s", job_id, spec.name)
        body = self._wait_for_job(job_id, spec.name)
        result = body.result
        if result is None:
            raise RuntimeError(f"Worker job {job_id} ended without result: {body}")
        samples, sample_rate = _decode_wav_b64(result.audio_b64)
        logger.info(
            "RemoteRenderer stem=%s duration=%.1fs (server=%.1fs)",
            spec.name,
            result.actual_duration_s,
            result.elapsed_s,
        )
        return AudioStem(
            spec=spec.model_copy(update={"seed": result.seed_used}),
            samples=samples,
            sample_rate=sample_rate,
            actual_duration_s=result.actual_duration_s,
        )

    def _submit_job(self, spec: StemSpec) -> str:
        payload = RemoteGenerateRequest(spec=spec).model_dump(mode="json")
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)
            ),
            reraise=True,
        ):
            with attempt, httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self._base_url}/jobs",
                    json=payload,
                    headers=self._headers(),
                )
                if resp.status_code >= 400:
                    detail = self._extract_error_detail(resp)
                    raise RuntimeError(
                        f"Worker returned HTTP {resp.status_code} on POST /jobs: {detail}"
                    )
                return RemoteJobAccepted.model_validate(resp.json()).job_id
        raise RuntimeError("unreachable")

    def _wait_for_job(self, job_id: str, stem_name: str) -> RemoteJobStatus:
        deadline = time.perf_counter() + self._max_wait_s
        consecutive_network_errors = 0
        while True:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"Job {job_id} did not finish within {self._max_wait_s:.0f}s."
                )
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(
                        f"{self._base_url}/jobs/{job_id}",
                        headers=self._headers(),
                    )
                consecutive_network_errors = 0
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                consecutive_network_errors += 1
                if consecutive_network_errors >= 6:
                    raise RuntimeError(
                        f"Lost connection to worker for job {job_id}: {e}"
                    ) from e
                logger.warning(
                    "Transient network error polling job %s (%d/6): %s",
                    job_id,
                    consecutive_network_errors,
                    e,
                )
                time.sleep(self._poll_interval_s)
                continue

            if resp.status_code >= 400:
                detail = self._extract_error_detail(resp)
                raise RuntimeError(
                    f"Worker returned HTTP {resp.status_code} on GET /jobs/{job_id}: {detail}"
                )
            body = RemoteJobStatus.model_validate(resp.json())
            if body.status == "done":
                return body
            if body.status == "error":
                raise RuntimeError(
                    f"Worker job {job_id} failed (stem={stem_name}): {body.error}"
                )
            logger.info(
                "job=%s stem=%s status=%s progress=%.0f%%",
                job_id,
                stem_name,
                body.status,
                body.progress * 100,
            )
            time.sleep(self._poll_interval_s)

    def health(self) -> dict[str, object]:
        """Probe ``GET /health`` on the worker. Returns the parsed JSON."""
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self._base_url}/health", headers=headers)
            resp.raise_for_status()
            return resp.json()

    def unload(self) -> None:
        return

    @staticmethod
    def _extract_error_detail(resp: httpx.Response) -> str:
        try:
            data = resp.json()
            if isinstance(data, dict) and "detail" in data:
                return str(data["detail"])
            return str(data)
        except Exception:
            text = resp.text or ""
            return text[:800] if text else "(empty body)"


def _decode_wav_b64(b64: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(b64)
    data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    samples = np.asarray(data, dtype=np.float32).T
    if samples.shape[0] == 1:
        samples = np.concatenate([samples, samples], axis=0)
    return samples, int(sample_rate)
