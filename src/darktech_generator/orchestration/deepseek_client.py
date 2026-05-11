"""Thin async client over DeepSeek's OpenAI-compatible Chat Completions endpoint.

Pydantic AI has providers for DeepSeek via its OpenAI-compatible interface, but we
keep a direct httpx wrapper for two reasons:

1. We need precise control over the JSON response_format and timeouts.
2. We track per-call token usage to feed :class:`CostTracker`.

The agent module uses this client when it needs structured JSON returns that the
Pydantic AI Agent path cannot give us deterministically.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from darktech_generator.config import get_settings
from darktech_generator.orchestration.cost_tracker import CostTracker
from darktech_generator.schemas import CostEntry


class DeepSeekClient:
    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        timeout_s: float = 90.0,
    ) -> None:
        self._settings = get_settings()
        api_key = self._settings.deepseek_api_key.get_secret_value()
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Configure it via .env or environment."
            )
        self._http = httpx.AsyncClient(
            base_url=self._settings.deepseek_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_s,
        )
        self._cost_tracker = cost_tracker or CostTracker()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "DeepSeekClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    async def chat_json(
        self,
        system: str,
        user: str,
        operation: str,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Issue a chat completion with response_format=json_object and parse the result."""
        payload = {
            "model": self._settings.deepseek_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                response = await self._http.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        usd = (
            input_tokens / 1_000_000 * self._settings.deepseek_price_input_per_mtoken_usd
            + output_tokens / 1_000_000 * self._settings.deepseek_price_output_per_mtoken_usd
        )
        self._cost_tracker.record(
            CostEntry(
                provider="deepseek",
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usd=usd,
            )
        )

        message = data["choices"][0]["message"]
        content = message.get("content", "")
        import json

        return json.loads(content)
