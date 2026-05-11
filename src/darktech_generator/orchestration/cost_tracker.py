"""In-memory cost tracker with a configurable hard cap per session."""

from __future__ import annotations

from threading import Lock

from darktech_generator.config import get_settings
from darktech_generator.schemas import CostEntry


class CostCapExceeded(RuntimeError):
    """Raised when the cumulative cost of a session crosses the hard cap."""


class CostTracker:
    def __init__(self, hard_cap_usd: float | None = None) -> None:
        self._lock = Lock()
        self._entries: list[CostEntry] = []
        self._hard_cap_usd = (
            hard_cap_usd
            if hard_cap_usd is not None
            else get_settings().darktech_cost_hard_cap_usd
        )

    def record(self, entry: CostEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if self.total_usd > self._hard_cap_usd:
                raise CostCapExceeded(
                    f"Session cost ${self.total_usd:.4f} exceeded hard cap ${self._hard_cap_usd:.2f}"
                )

    @property
    def total_usd(self) -> float:
        return sum(e.usd for e in self._entries)

    @property
    def entries(self) -> list[CostEntry]:
        return list(self._entries)
