"""Top-level orchestrator: maps user prompt + optional references to a TrackPlan.

For the MVP this is a single node (plan_track). Phase 2 adds plan_stems, critique,
and refine_prompt nodes; the public surface (`OrchestratorAgent.plan`) stays stable.
"""

from __future__ import annotations

import json
from pathlib import Path

from darktech_generator.orchestration.deepseek_client import DeepSeekClient
from darktech_generator.schemas import ReferenceProfile, TrackPlan

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_system_prompt() -> str:
    director = (PROMPTS_DIR / "system_director.md").read_text(encoding="utf-8")
    taxonomy = (PROMPTS_DIR / "darktech_taxonomy.md").read_text(encoding="utf-8")
    few_shot = json.loads((PROMPTS_DIR / "few_shot_examples.json").read_text(encoding="utf-8"))
    few_shot_block = "\n\n## Few-shot examples\n\n"
    for ex in few_shot:
        few_shot_block += f"USER: {ex['user_prompt']}\n"
        few_shot_block += f"ASSISTANT (JSON):\n{json.dumps(ex['expected_output'], indent=2)}\n\n"
    return f"{director}\n\n---\n\n{taxonomy}\n\n---\n{few_shot_block}"


class OrchestratorAgent:
    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self._client = client
        self._system_prompt = _load_system_prompt()

    async def __aenter__(self) -> "OrchestratorAgent":
        if self._client is None:
            self._client = DeepSeekClient()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()

    @property
    def client(self) -> DeepSeekClient:
        if self._client is None:
            raise RuntimeError("OrchestratorAgent used outside of async context manager.")
        return self._client

    async def plan(
        self,
        user_prompt: str,
        reference: ReferenceProfile | None = None,
    ) -> TrackPlan:
        """Produce a TrackPlan from the user's natural-language brief."""
        ref_block = reference.to_llm_summary() if reference else "No reference tracks provided."
        user_message = (
            f"USER BRIEF:\n{user_prompt}\n\n"
            f"REFERENCE PROFILE:\n{ref_block}\n\n"
            "Respond with a JSON object validating against the TrackPlan schema. "
            "Do not include `stem_specs` for now; leave it as an empty array."
        )
        raw = await self.client.chat_json(
            system=self._system_prompt,
            user=user_message,
            operation="plan_track",
            temperature=0.4,
            max_tokens=2048,
        )
        return TrackPlan.model_validate(raw)
