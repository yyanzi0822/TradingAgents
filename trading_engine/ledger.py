"""Append-only, hash-chained ledger of decisions, risk results and orders.

Every record carries the hash of the record before it, so editing or deleting
an earlier line breaks ``verify()``. The decision is written *before* any
order is sent, which is what makes the forward test auditable: you can prove
what the agent decided before the market answered.

Also stores the equity high-water mark used by the drawdown breaker, since
Alpaca does not report one.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(prev_hash: str, body: dict) -> str:
    payload = prev_hash + json.dumps(body, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def last_hash(self) -> str:
        records = self._records()
        return records[-1]["hash"] if records else GENESIS

    def append(self, kind: str, data: dict[str, Any]) -> dict:
        body = {
            "kind": kind,
            "at": datetime.now(UTC).isoformat(),
            "data": data,
        }
        prev = self.last_hash()
        record = {**body, "prev_hash": prev, "hash": _digest(prev, body)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, rec in enumerate(self._records(), start=1):
            body = {k: rec[k] for k in ("kind", "at", "data")}
            if rec.get("prev_hash") != prev or rec.get("hash") != _digest(prev, body):
                return False, f"chain broken at record {i}"
            prev = rec["hash"]
        return True, "ok"

    def records(self, kind: str | None = None) -> list[dict]:
        return [r for r in self._records() if kind is None or r["kind"] == kind]

    def high_water_mark(self) -> float | None:
        marks = [r["data"]["equity"] for r in self.records("equity")]
        return max(marks) if marks else None

    def submitted_order_ids(self) -> set[str]:
        return {r["data"]["client_order_id"] for r in self.records("order")}
