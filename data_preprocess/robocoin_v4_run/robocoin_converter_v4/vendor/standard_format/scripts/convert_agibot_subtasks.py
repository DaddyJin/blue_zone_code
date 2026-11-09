#!/usr/bin/env python
"""Convert AgiBot lerobot v3 (canonical_55d) subtask annotations to the
EgoStandard-55d ("050") storage/index convention.

Per task directory (<task>/canonical_55d):
  1. meta/episodes.jsonl `action_config` (per-episode list of
     {action_text, start_frame, end_frame, skill}, episode-relative frames,
     contiguous segments) is turned into a per-frame int64 column
     `subtask_index` appended to every data/*.parquet file.
     Frames outside [first.start_frame, last.end_frame) get -1, matching
     EgoStandard's "unlabeled frame" convention.
  2. meta/subtasks.parquet is written as the subtask vocabulary
     (columns: subtask_index int64, subtask string; pandas index = subtask),
     byte-layout identical to EgoStandard's.
  3. meta/info.json gains `features.subtask_index = {dtype: int64, shape: [1]}`.
  4. episodes.jsonl (with action_config incl. skill) is kept untouched as a
     sidecar, same as EgoStandard.

Idempotent: tasks with a `.subtask050` marker are skipped. Data parquet
rewrite is atomic (tmp file + rename). Resume-safe to re-run any time.

Usage:
  convert_agibot_subtasks.py <agibot_root> [--workers 8] [--tasks task_352,...]

<agibot_root> is the dir that directly contains task_* dirs, e.g.
/mnt/disk2/agibot_lerobot_v3/agibot  (each task dir holds canonical_55d/).

Requires: pyarrow, numpy, pandas (tested with the forge_venv interpreter).
Log lines follow the task-monitor generic format:
  TASK_START total=N / [i/N] <task> OK|FAIL|SKIP <sec>s / TASK_DONE
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MARKER = ".subtask050"


def log(line: str) -> None:
    print(line, flush=True)


def build_vocab(episodes: list[dict]) -> list[str]:
    """First-seen ordered unique action_text across all episodes."""
    seen: dict[str, None] = {}
    for ep in episodes:
        for seg in ep.get("action_config") or []:
            seen.setdefault(seg["action_text"])
    return list(seen)


def episode_lut(ep: dict, vocab_pos: dict[str, int]):
    """Return (ends int64 array, vocab_idx int64 array) for one episode.

    ends[k] = end_frame of k-th segment (exclusive); vocab_idx[k] = global
    subtask index of that segment. Frames map via searchsorted(ends, f,
    side='right'); result == len(ends) means past the last segment -> -1.
    Frames before the first start also land outside segments only when
    start_frame > 0 (head gap); searchsorted handles that only if we also
    track starts, so we return starts too.
    """
    cfg = ep.get("action_config") or []
    starts = np.array([s["start_frame"] for s in cfg], dtype=np.int64)
    ends = np.array([s["end_frame"] for s in cfg], dtype=np.int64)
    vidx = np.array([vocab_pos[s["action_text"]] for s in cfg], dtype=np.int64)
    return starts, ends, vidx


def convert_task(task_dir: Path) -> dict:
    canon = task_dir / "canonical_55d"
    marker = canon / MARKER
    if marker.exists():
        return {"task": task_dir.name, "status": "SKIP", "sec": 0.0}
    t0 = time.perf_counter()

    meta = canon / "meta"
    episodes = [json.loads(l) for l in open(meta / "episodes.jsonl")]
    vocab = build_vocab(episodes)
    if not vocab:
        raise RuntimeError("no action_config found in episodes.jsonl")
    vocab_pos = {t: i for i, t in enumerate(vocab)}
    lut = {ep["episode_index"]: episode_lut(ep, vocab_pos) for ep in episodes}

    stats = {"frames": 0, "head_unlabeled": 0, "tail_unlabeled": 0}

    data_files = sorted((canon / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise RuntimeError("no data parquet files")
    for df in data_files:
        # ParquetFile.read(): the dataset scanner behind pq.read_table chokes
        # on duplicate column names left by an interrupted earlier run
        tbl = pq.ParquetFile(df).read()
        if "subtask_index" in tbl.column_names:
            # re-run after a partial attempt: drop stale column(s) by position
            tbl = tbl.select([i for i, n in enumerate(tbl.column_names)
                              if n != "subtask_index"])
        ep_arr = tbl.column("episode_index").to_numpy()
        fr_arr = tbl.column("frame_index").to_numpy()
        sub = np.full(len(fr_arr), -1, dtype=np.int64)
        # rows are ordered by episode; process per unique episode slice
        change = np.flatnonzero(np.diff(ep_arr)) + 1
        bounds = np.concatenate(([0], change, [len(ep_arr)]))
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            starts, ends, vidx = lut[int(ep_arr[lo])]
            f = fr_arr[lo:hi]
            pos = np.searchsorted(ends, f, side="right")      # segment k: ends[k-1] <= f < ends[k]
            inside = pos < len(ends)
            # exclude frames before the segment's start (head gap / holes)
            if len(starts):
                inside &= f >= starts[np.clip(pos, 0, len(starts) - 1)]
            mapped = np.where(inside, vidx[np.clip(pos, 0, len(vidx) - 1)], -1)
            sub[lo:hi] = mapped
            stats["frames"] += hi - lo
            stats["head_unlabeled"] += int(np.sum((f < (starts[0] if len(starts) else 0)) & (sub[lo:hi] == -1)))
            stats["tail_unlabeled"] += int(np.sum((f >= (ends[-1] if len(ends) else 0)) & (sub[lo:hi] == -1)))
        # insert subtask_index right before the trailing `index` column
        names = tbl.column_names
        pos_idx = names.index("index") if "index" in names else len(names)
        new_tbl = tbl.add_column(pos_idx, "subtask_index", pa.array(sub, type=pa.int64()))
        tmp = df.with_suffix(".parquet.tmp050")
        pq.write_table(new_tbl, tmp, compression="zstd")
        tmp.rename(df)

    # subtasks.parquet: same pandas-index layout as EgoStandard
    import pandas as pd
    sdf = pd.DataFrame({"subtask_index": range(len(vocab))},
                       index=pd.Index(vocab, name="subtask"))
    sdf.to_parquet(meta / "subtasks.parquet", compression="zstd")

    # info.json: register the feature
    info_path = meta / "info.json"
    info = json.loads(info_path.read_text())
    feats = info.setdefault("features", {})
    if "subtask_index" not in feats:
        # insert after reliability if present, else append
        items = list(feats.items())
        feats.clear()
        inserted = False
        for k, v in items:
            feats[k] = v
            if k == "reliability":
                feats["subtask_index"] = {"dtype": "int64", "shape": [1]}
                inserted = True
        if not inserted:
            feats["subtask_index"] = {"dtype": "int64", "shape": [1]}
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))

    sec = time.perf_counter() - t0
    stats = {k: int(v) for k, v in stats.items()}
    marker.write_text(json.dumps({"vocab": len(vocab), **stats}) + "\n")
    return {"task": task_dir.name, "status": "OK", "sec": sec,
            "vocab": len(vocab), **stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("agibot_root", help="dir containing task_* dirs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", default=None, help="comma-separated subset")
    args = ap.parse_args()

    root = Path(args.agibot_root)
    names = sorted(p.name for p in root.iterdir()
                   if p.is_dir() and (p / "canonical_55d").is_dir())
    if args.tasks:
        keep = set(args.tasks.split(","))
        names = [n for n in names if n in keep]
    log(f"TASK_START total={len(names)}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(convert_task, root / n): (i, n)
                for i, n in enumerate(names, 1)}
        for fut, (i, n) in futs.items():
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = {"task": n, "status": "FAIL", "sec": 0.0, "err": str(e)}
            results.append(r)
            extra = (f" frames={r['frames']} head-={r['head_unlabeled']}"
                     f" tail-={r['tail_unlabeled']}" if r["status"] == "OK"
                     else r.get("err", ""))
            log(f"[{i}/{len(names)}] {n} {r['status']} {r['sec']:.1f}s {extra}".rstrip())

    fails = [r for r in results if r["status"] == "FAIL"]
    log("TASK_DONE")
    (root / "subtask050_summary.json").write_text(json.dumps(results, indent=1))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
