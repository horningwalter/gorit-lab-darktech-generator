"""HTTP client that delegates rendering to a remote Colab worker.

Implements the :class:`StemRenderer` protocol so the rest of the pipeline cannot
tell whether generation is happening locally on a GPU or remotely on a Colab
A100. The worker side lives in :mod:`darktech_generator.generation.remote_worker`.

Wire format: base64-encoded WAV bytes in the JSON response. Authentication via
``X-API-Key`` header. Retries with exponential backoff on network/5xx errors.
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
    RemoteGenerateResponse,
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
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.darktech_remote_url).rstrip("/")
        self._api_key = api_key or settings.darktech_remote_api_key.get_secret_value()
        self._timeout_s = timeout_s or settings.darktech_remote_timeout_s
        if not self._base_url:
            raise RuntimeError(
                "RemoteRenderer requires DARKTECH_REMOTE_URL to be set."
            )

    def render(self, spec: StemSpec) -> AudioStem:
        payload = RemoteGenerateRequest(spec=spec).model_dump(mode="json")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        started = time.perf_counter()
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)),
            reraise=True,
        ):
            with attempt, httpx.Client(timeout=self._timeout_s) as client:
                logger.info("RemoteRenderer.render -> %s (stem=%s)", self._base_url, spec.name)
                resp = client.post(
                    f"{self._base_url}/generate",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code >= 400:
                    detail = self._extract_error_detail(resp)
                    raise RuntimeError(
                        f"Remote worker returned HTTP {resp.status_code}: {detail}"
                    )
                body = RemoteGenerateResponse.model_validate(resp.json())

        samples, sample_rate = _decode_wav_b64(body.audio_b64)
        elapsed_total = time.perf_counter() - started
        logger.info(
            "RemoteRenderer stem=%s duration=%.1fs (server=%.1fs, total=%.1fs)",
            spec.name,
            body.actual_duration_s,
            body.elapsed_s,
            elapsed_total,
        )
        return AudioStem(
            spec=spec.model_copy(update={"seed": body.seed_used}),
            samples=samples,
            sample_rate=sample_rate,
            actual_duration_s=body.actual_duration_s,
        )

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
