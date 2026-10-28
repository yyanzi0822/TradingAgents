"""File-based emergency stop.

While the kill-switch file exists, the risk engine rejects every order. It is a
file (not a setting in code) so you can flip it from a shell, a cron job or a
monitoring alert without touching the running process.

    python -m trading_engine.killswitch on "reason"
    python -m trading_engine.killswitch off
    python -m trading_engine.killswitch status
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine")) / "KILL_SWITCH"


class KillSwitch:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)

    def active(self) -> bool:
        return self.path.exists()

    def reason(self) -> str | None:
        return self.path.read_text(encoding="utf-8").strip() if self.active() else None

    def engage(self, reason: str = "manual") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{datetime.now(UTC).isoformat()} {reason}\n", encoding="utf-8")

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    ks = KillSwitch()
    cmd = argv[0] if argv else "status"
    if cmd == "on":
        ks.engage(" ".join(argv[1:]) or "manual")
    elif cmd == "off":
        ks.release()
    elif cmd != "status":
        print(__doc__)
        return 2
    print(f"kill switch: {'ON — ' + (ks.reason() or '') if ks.active() else 'off'} ({ks.path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
