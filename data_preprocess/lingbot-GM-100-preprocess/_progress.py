#!/usr/bin/env python
"""Tiny progress/ETA reporter (stdlib only), shared by the preprocess scripts.

Deliberately dependency-free (no tqdm) so each project stays self-contained:
copy this file together with the other scripts. The same file is vendored in
each project directory.

Design:
  - counts are weighted units (frames for convert/validate, archives for
    extract), so ETA is meaningful when items differ in size;
  - writes a line every `interval` seconds to non-TTY logs (nohup/tail -f) and
    refreshes in place on a TTY;
  - a daemon heartbeat keeps printing during long single-item work (e.g. one
    ffmpeg transcode) even when update() is not called frequently;
  - update() is thread-safe: parallel video workers call it concurrently.
"""
from __future__ import annotations

import queue as queue_mod
import shutil
import sys
import threading
import time
from typing import TextIO

# serialize in-place progress refreshes with ordinary log lines (see log_line)
_WRITE_LOCK = threading.RLock()


def log_line(msg: str = "", stream: TextIO | None = None) -> None:
    """print()-replacement that keeps the in-place progress bar intact.

    On a TTY the progress line is drawn without a trailing newline; a plain
    print() would append to it and leave a broken row. This clears the line
    (CR + erase-to-end) first, under the same lock as the progress writes.
    On non-TTY streams (logs / pipes) this is equivalent to print().
    """
    out = stream if stream is not None else sys.stdout
    try:
        is_tty = bool(getattr(out, "isatty", lambda: False)())
    except Exception:  # noqa: BLE001
        is_tty = False
    with _WRITE_LOCK:
        if is_tty:
            out.write("\r\x1b[K")
        out.write(msg)
        if not msg.endswith("\n"):
            out.write("\n")
        out.flush()


def fmt_duration(seconds: float | None) -> str:
    """Seconds -> HH:MM:SS (or --:--:-- when unknown)."""
    if seconds is None or seconds != seconds:  # None / NaN
        return "--:--:--"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Progress:
    """Thread-safe progress + ETA reporter.

    Example:
        p = Progress(total=1000, unit="frames", label="convert")
        p.start_heartbeat()
        p.update(100, note="task_00001")
        p.finish()
    """

    def __init__(self, total: int = 0, unit: str = "items", label: str = "",
                 enabled: bool = True, interval: float = 10.0,
                 stream: TextIO | None = None):
        self.total = max(0, int(total or 0))
        self.unit = unit
        self.label = label
        self.enabled = enabled
        self.interval = max(0.5, float(interval))
        self.stream = stream if stream is not None else sys.stdout
        self.done = 0.0
        self.rate = 0.0                     # smoothed units/second
        self.start = time.monotonic()
        self._last_print = 0.0
        self._last_t = self.start
        self._last_done = 0
        self._note = ""
        self._lock = threading.RLock()
        self._finished = False
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None

    # ------------------------------------------------------------------ state
    def set_total(self, total: int, unit: str | None = None) -> None:
        with self._lock:
            self.total = max(0, int(total or 0))
            if unit:
                self.unit = unit

    def update(self, n: float = 1, note: str | None = None) -> None:
        with self._lock:
            self.done += float(n)
            if note is not None:
                self._note = note
        self.maybe_print()

    def set_note(self, note: str) -> None:
        with self._lock:
            self._note = note

    # ----------------------------------------------------------------- render
    def _tick_rate(self, now: float) -> None:
        dd = self.done - self._last_done
        dt = now - self._last_t
        if dt >= 0.5 and dd >= 0:
            inst = dd / dt
            if inst > 0:
                self.rate = inst if self.rate <= 0 else 0.3 * inst + 0.7 * self.rate
            self._last_t = now
            self._last_done = self.done

    def _render(self, now: float) -> str:
        elapsed = now - self.start
        parts = [f"[PROGRESS] {self.label}".rstrip()]
        done = int(round(self.done))
        if self.total > 0:
            pct = 100.0 * self.done / self.total
            parts.append(f"{done:,}/{self.total:,} {self.unit} ({pct:.1f}%)")
            eta = (self.total - self.done) / self.rate if self.rate > 0 else None
        else:
            parts.append(f"{done:,} {self.unit}")
            eta = None
        parts.append(f"elapsed {fmt_duration(elapsed)}")
        if eta is not None:
            parts.append(f"ETA {fmt_duration(eta)}")
        if self.rate > 0:
            parts.append(f"{self.rate:,.1f} {self.unit}/s")
        if self._note:
            parts.append(f"| {self._note}")
        return " ".join(parts)

    def maybe_print(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            if self._finished:
                return
            if not force and now - self._last_print < self.interval:
                return
            self._tick_rate(now)
            line = self._render(now)
            self._last_print = now
            self._write(line)
        self.stream.flush()

    def _term_width(self) -> int:
        """Usable terminal width (minus 1 col so a full line never wraps)."""
        try:
            cols = shutil.get_terminal_size((120, 20)).columns
        except Exception:  # noqa: BLE001  (defensive: no tty info)
            cols = 120
        return max(20, cols - 1)

    def _write(self, line: str) -> None:
        with _WRITE_LOCK:
            if self._is_tty:
                # Single-line in-place refresh: CR + erase-to-end-of-line. Never
                # pad to a fixed width -- a line wider than the terminal wraps
                # and CR would then only rewind the last visual row.
                w = self._term_width()
                if len(line) > w:
                    line = line[:w - 1] + "…"
                self.stream.write("\r" + line + "\x1b[K")
            else:
                self.stream.write(line + "\n")

    # -------------------------------------------------------------- lifecycle
    def start_heartbeat(self) -> None:
        """Print periodically even when no update() calls arrive (long ffmpeg)."""
        if not self.enabled or self._heartbeat is not None:
            return

        def _loop() -> None:
            while not self._stop.wait(1.0):
                self.maybe_print()

        self._heartbeat = threading.Thread(
            target=_loop, daemon=True, name=f"progress-{self.label or 'task'}")
        self._heartbeat.start()

    def finish(self, note: str | None = None, ok: bool = True) -> None:
        if not self.enabled:
            self._stop.set()
            return
        self._stop.set()
        if note is not None:
            self._note = note
        now = time.monotonic()
        with self._lock:
            if self._finished:
                return
            self._tick_rate(now)
            line = self._render(now)
            if not ok:
                line += " | INCOMPLETE"
            with _WRITE_LOCK:
                if self._is_tty:
                    w = self._term_width()
                    if len(line) > w:
                        line = line[:w - 1] + "…"
                    self.stream.write("\r" + line + "\x1b[K\n")
                else:
                    self.stream.write(line + "\n")
                self.stream.flush()
            self._finished = True


def start_queue_forwarder(q, prog: Progress,
                          stop_event: threading.Event) -> threading.Thread:
    """Forward (n, note) tuples from a multiprocessing queue into `prog`.

    Used by the parallel batch mode: worker processes report their in-dataset
    progress through a Manager queue, the parent prints one aggregated ETA.
    Send `None` (or set stop_event and let the queue drain) to stop the thread.
    """
    def _loop() -> None:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue_mod.Empty:
                if stop_event.is_set():
                    break
                continue
            if item is None:
                break
            n, note = item
            prog.update(float(n), note=str(note))

    t = threading.Thread(target=_loop, daemon=True, name="progress-queue")
    t.start()
    return t
