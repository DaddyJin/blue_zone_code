#!/usr/bin/env python
"""Repair the subtask_index boundary-frame off-by-one in already-converted
egostandard_55d tasks.

Background: egostandard_staging.py originally computed the per-frame subtask
column by bisecting end_times directly, while action_config bounds used
bisect_right(timestamps, et). At frames with t == end_time (every segment
boundary — annotation end_times coincide with frame timestamps) the per-frame
column assigned the NEXT segment, and frames past the last end_time got -1.
action_config (agibot-compatible) is authoritative: segment k covers
[start_frame, end_frame).

This script rebuilds the expected per-frame subtask_index from
meta/episodes.jsonl action_config + meta/subtasks.parquet vocab, compares
against the stored column, and rewrites only the data files that differ
(tmp+rename; never in-place on hardlinks).

Idempotent: a second run fixes nothing. Run AFTER the p2 merge driver has
finished (it rewrites whole data files, so it must not race a merge).

Usage:
  egostandard_fix_subtask_bounds.py /mnt/disk2/egostandard_55d [--workers 4]
Prints task-monitor generic log lines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def log(line: str) -> None:
    print(line, flush=True)


def fix_task(task: Path) -> tuple[int, int]:
    """Returns (files_rewritten, frames_fixed)."""
    side = [json.loads(l) for l in open(task / "meta" / "episodes.jsonl", encoding="utf-8")]
    ac = {d["episode_index"]: d.get("action_config") or [] for d in side}
    subs = sorted(pq.read_table(task / "meta" / "subtasks.parquet").to_pylist(),
                  key=lambda r: r["subtask_index"])
    vocab = [r["subtask"] for r in subs]

    # action_config is the source of truth. Old conversions could miss vocab
    # entries for segments that never owned a frame (e.g. zero-width segments
    # from non-monotonic source end_times): append them, old indices stable.
    extra = []
    for segs in ac.values():
        for seg in segs:
            t = seg["action_text"] or None
            if t is not None and t not in vocab and t not in extra:
                extra.append(t)
    if extra:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from egostandard_merge_p2 import _write_vocab
        _write_vocab(task / "meta" / "subtasks.parquet", "subtask", vocab + extra)
        vocab = vocab + extra
        log(f"  {task.name}: vocab extended +{len(extra)}")
    idx_of = {t: i for i, t in enumerate(vocab)}

    fixed_files = fixed_frames = 0
    for f in sorted((task / "data" / "chunk-000").glob("file-*.parquet")):
        t = pq.read_table(f)
        eps = t["episode_index"].to_pylist()
        fis = t["frame_index"].to_pylist()
        sis = t["subtask_index"].to_pylist()
        exp = []
        for e, fi in zip(eps, fis):
            txt = None
            for seg in ac.get(e, []):
                if seg["start_frame"] <= fi < seg["end_frame"]:
                    txt = seg["action_text"] or None
                    break
            exp.append(-1 if txt is None else idx_of[txt])
        if exp != sis:
            t = t.set_column(t.schema.get_field_index("subtask_index"), "subtask_index",
                             pa.array(exp, type=pa.int64()))
            tmp = f.with_suffix(".parquet.tmp")
            pq.write_table(t, tmp, compression="zstd")
            tmp.rename(f)
            fixed_files += 1
            fixed_frames += sum(1 for a, b in zip(exp, sis) if a != b)
    return fixed_files, fixed_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task dir names")
    args = ap.parse_args()

    tasks = sorted(d for d in Path(args.root).iterdir()
                   if d.is_dir() and (d / "meta" / "info.json").exists())
    if args.tasks:
        tasks = [d for d in tasks if d.name in set(args.tasks)]
    log(f"TASK_START total={len(tasks)}")

    def work(item):
        i, task = item
        t0 = time.perf_counter()
        try:
            nf, nfr = fix_task(task)
            log(f"[{i}/{len(tasks)}] {task.name} OK {time.perf_counter() - t0:.1f}s "
                f"files_rewritten={nf} frames_fixed={nfr}")
        except Exception as e:  # noqa: BLE001
            log(f"[{i}/{len(tasks)}] {task.name} FAIL {time.perf_counter() - t0:.1f}s {e}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, enumerate(tasks, 1)))
    log("TASK_DONE")


if __name__ == "__main__":
    sys.exit(main())
