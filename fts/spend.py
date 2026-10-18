"""Hosted-model spend ledger with a hard Spend_Cap (Req 3.3, 3.4).

Every hosted request in the project is appended to ``results/hosted_ledger.jsonl``.
Cost per request is ``prompt_tokens * prompt_price + completion_tokens * completion_price``
(prices per 1M tokens). The prompt price is the higher of the input and
cache-write rates, so the recorded spend is an upper bound on the bill.

Before each request the caller asks :meth:`SpendLedger.check` whether the
request's worst case (prompt length estimated from its characters, plus the full
output budget) still fits under the cap. If not, :class:`SpendCapExceeded` is
raised and the request is never sent.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: Characters per token used for the worst-case prompt estimate (design.md: characters / 3).
CHARS_PER_TOKEN = 3
#: Extra tokens per chat message for role and formatting markers, added to the estimate.
TOKENS_PER_MESSAGE = 8


class SpendCapExceeded(RuntimeError):
    """A request's worst-case cost would push recorded spend above the Spend_Cap."""

    def __init__(self, spent_usd: float, cap_usd: float, worst_case_usd: float) -> None:
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        self.worst_case_usd = worst_case_usd
        super().__init__(
            f"Spend_Cap reached: spent ${spent_usd:.6f} of the ${cap_usd:.2f} cap; the next request "
            f"could cost up to ${worst_case_usd:.6f}, so it was not sent"
        )


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""

    prompt_per_mtok: float
    completion_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.prompt_per_mtok + completion_tokens * self.completion_per_mtok) / 1_000_000


def prices_from_config(cfg: dict) -> Prices:
    """Prompt tokens at max(input, cache_write); completion tokens at the output rate."""
    p = cfg["prices_usd_per_1m_tokens"]
    return Prices(prompt_per_mtok=max(float(p["input"]), float(p["cache_write"])), completion_per_mtok=float(p["output"]))


def estimate_prompt_tokens(messages: list[dict]) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return math.ceil(chars / CHARS_PER_TOKEN) + TOKENS_PER_MESSAGE * len(messages)


class SpendLedger:
    def __init__(self, path: Path | str, cap_usd: float, prices: Prices, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        self.prices = prices
        self._clock = clock
        self.entries: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))

    @property
    def total_usd(self) -> float:
        return sum(self.prices.cost(e["prompt_tokens"], e["completion_tokens"]) for e in self.entries)

    @property
    def requests(self) -> int:
        return len(self.entries)

    def worst_case_usd(self, prompt_tokens: int, max_completion_tokens: int) -> float:
        return self.prices.cost(prompt_tokens, max_completion_tokens)

    def check_tokens(self, prompt_tokens: int, max_completion_tokens: int) -> None:
        """Raise SpendCapExceeded unless spent + worst case <= cap."""
        worst = self.worst_case_usd(prompt_tokens, max_completion_tokens)
        spent = self.total_usd
        if spent + worst > self.cap_usd:
            raise SpendCapExceeded(spent, self.cap_usd, worst)

    def check(self, messages: list[dict], max_completion_tokens: int) -> None:
        self.check_tokens(estimate_prompt_tokens(messages), max_completion_tokens)

    def record(self, prompt_tokens: int, completion_tokens: int, model: str, tag: str = "") -> dict:
        """Append one completed request; returns the entry written."""
        entry = {
            "time": round(self._clock(), 3),
            "tag": tag,
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cost_usd": self.prices.cost(prompt_tokens, completion_tokens),
        }
        self.entries.append(entry)
        entry_with_total = dict(entry, total_usd=self.total_usd)
        self.entries[-1] = entry_with_total
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry_with_total, sort_keys=True) + "\n")
        return entry_with_total

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "prompt_tokens": sum(e["prompt_tokens"] for e in self.entries),
            "completion_tokens": sum(e["completion_tokens"] for e in self.entries),
            "spend_usd": self.total_usd,
            "spend_cap_usd": self.cap_usd,
            "prompt_price_per_mtok": self.prices.prompt_per_mtok,
            "completion_price_per_mtok": self.prices.completion_per_mtok,
        }
