"""Count model calls and tokens per ticker, so every cycle records what it cost.

A ``UsageCounter`` is attached to the analysis graph as a LangChain callback.
The analyzer reads it around each ticker: ``start()`` before the run,
``snapshot()`` after, and the runner writes the result to the ledger as a
``usage`` record.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(frozen=True)
class Usage:
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def as_dict(self) -> dict:
        return {**asdict(self), "tokens": self.tokens}

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.llm_calls + other.llm_calls, self.tokens_in + other.tokens_in,
                     self.tokens_out + other.tokens_out, self.seconds + other.seconds)


class UsageCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.start()

    def start(self) -> None:
        with self._lock:
            self.llm_calls = self.tokens_in = self.tokens_out = 0
            self._t0 = time.monotonic()

    def snapshot(self) -> Usage:
        with self._lock:
            return Usage(self.llm_calls, self.tokens_in, self.tokens_out,
                         round(time.monotonic() - self._t0, 1))

    # LangChain callbacks ------------------------------------------------------

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list, **kwargs: Any) -> None:
        with self._lock:
            self.llm_calls += 1

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError, TypeError):
            return
        meta = getattr(message, "usage_metadata", None) or {}
        with self._lock:
            self.tokens_in += int(meta.get("input_tokens", 0) or 0)
            self.tokens_out += int(meta.get("output_tokens", 0) or 0)


def fmt_tokens(n: int) -> str:
    """176400 -> '17.6万'; for the Chinese report."""
    return f"{n / 10_000:.1f}万" if n >= 10_000 else str(n)
