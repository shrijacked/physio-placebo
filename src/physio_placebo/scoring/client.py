"""Logprob client protocol. Tests use ScriptedClient; the GPU uses vLLM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class LogprobClient(Protocol):
    def score_prompts(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        """Return per-prompt {token: logprob} tables for the label set."""


class ConstantClient:
    """Test / dry-run double: the same table for every prompt."""

    def __init__(self, table: dict[str, float] | None = None):
        self.table = table or {"A": -0.2, "B": -1.5}

    def score_prompts(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        return [dict(self.table) for _ in prompts]


class ScriptedClient:
    """Deterministic test double. One table per call, consumed in order."""

    def __init__(self, tables: Sequence[dict[str, float]]):
        self._tables = list(tables)
        self.calls: list[tuple[str, ...]] = []

    def score_prompts(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        self.calls.append(tuple(prompts))
        n = len(prompts)
        if len(self._tables) < n:
            raise AssertionError(f"ScriptedClient needs {n} tables, has {len(self._tables)}")
        out = self._tables[:n]
        self._tables = self._tables[n:]
        return out
