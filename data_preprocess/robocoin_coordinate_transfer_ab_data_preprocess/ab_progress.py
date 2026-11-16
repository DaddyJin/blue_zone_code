"""Small dependency-free progress line with rate and ETA."""
from __future__ import annotations

import sys
import time
from typing import Optional, TextIO


def format_duration(seconds: float) -> str:
    if seconds is None or not (seconds == seconds) or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Progress:
    """Progress line for a known total.

    TTY: update in place with ``\\r``.
    Non-TTY: print at most once every ``log_interval`` seconds.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str = "frames",
        stream: Optional[TextIO] = None,
        log_interval: float = 10.0,
        width: int = 28,
    ):
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.stream = stream if stream is not None else sys.stderr
        self.log_interval = log_interval
        self.width = width
        self.start = time.monotonic()
        self.last_log = 0.0
        self.last_n = 0
        self.last_t = self.start
        self.rate = 0.0
        self.n = 0
        self.closed = False
        self._complete_rendered = False
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._render()

    def update(self, n: int = 1, suffix: str = "") -> None:
        self.n = min(self.total, self.n + int(n))
        now = time.monotonic()
        dt = now - self.last_t
        if dt >= 0.5:
            inst = (self.n - self.last_n) / dt
            if inst > 0:
                self.rate = inst if self.rate == 0 else 0.85 * self.rate + 0.15 * inst
            self.last_n = self.n
            self.last_t = now
        if self.is_tty:
            self._render(suffix=suffix)
        elif now - self.last_log >= self.log_interval or self.n >= self.total:
            self._render(suffix=suffix, newline=True)
            self.last_log = now
            if self.n >= self.total:
                self._complete_rendered = True

    def set_total(self, total: int) -> None:
        self.total = max(int(total), 0)
        self.n = min(self.n, self.total)

    def _render(self, suffix: str = "", newline: bool = False) -> None:
        elapsed = time.monotonic() - self.start
        if self.total > 0:
            frac = min(1.0, self.n / self.total)
            filled = int(round(self.width * frac))
            bar = "#" * filled + "-" * (self.width - filled)
            pct = f"{frac * 100:6.2f}%"
            counts = f"{self.n:,}/{self.total:,}"
        else:
            bar = "-" * self.width
            pct = "  0.00%"
            counts = "0/0"
        if self.rate > 0 and self.n < self.total:
            eta = (self.total - self.n) / self.rate
            eta_s = format_duration(eta)
            rate_s = f"{self.rate:,.0f} {self.unit}/s"
        else:
            eta_s = "00:00:00" if self.n >= self.total and self.total > 0 else "--:--:--"
            rate_s = f"{self.rate:,.0f} {self.unit}/s" if self.rate > 0 else "-"
        line = (
            f"[{bar}] {pct} {self.desc} "
            f"{counts} {self.unit} | {rate_s} | "
            f"elapsed {format_duration(elapsed)} | ETA {eta_s}"
        )
        if suffix:
            line += f" | {suffix}"
        if newline or not self.is_tty:
            self.stream.write(line + "\n")
        else:
            # Clear the remainder of the terminal line.
            self.stream.write("\r" + line + "\x1b[K")
        self.stream.flush()

    def close(self, suffix: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        if self.n < self.total:
            self.n = self.total
        if self.is_tty:
            self._render(suffix=suffix, newline=True)
        elif not self._complete_rendered:
            self._render(suffix=suffix, newline=True)
