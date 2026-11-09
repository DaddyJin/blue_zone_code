#!/usr/bin/env python
"""Validate the subtask050 conversion of AgiBot lerobot v3 tasks.

Checks per task (canonical_55d):
  1. data parquet has int64 `subtask_index`; info.json registers the feature
  2. meta/subtasks.parquet has EgoStandard layout (subtask_index int64,
     subtask string as pandas index)
  3. EXACT per-frame correctness: for every episode, rebuild the expected
     subtask_index array from episodes.jsonl action_config and compare with
     the data parquet values element-wise (all frames, not sampled)
  4. text round-trip: subtasks.parquet maps every used index back to the
     action_text that episode's action_config declares

Usage:
  validate_agibot_subtasks.py <agibot_root> [--tasks task_352,...] [--official]
  --official additionally loads each task with the official LeRobotDataset
  (run that mode with the lerobot-env python and HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# NOTE: numpy/pyarrow/pandas are imported lazily inside validate_task.
# Importing pyarrow before lerobot breaks torchcodec's native library load
# in the lerobot env (symbol clash), so --official mode must import
# LeRobotDataset first.
if "--official" in sys.argv:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def expected_subtask_array(ep: dict, vocab_pos: dict[str, int]):
    import numpy as np
    L = ep["length"]
    out = np.full(L, -1, dtype=np.int64)
    for seg in ep.get("action_config") or []:
        out[seg["start_frame"]:seg["end_frame"]] = vocab_pos[seg["action_text"]]
    return out


def validate_task(task_dir: Path, official: bool) -> list[str]:
    import numpy as np
    import pyarrow.parquet as pq
    errs = []
    canon = task_dir / "canonical_55d"
    meta = canon / "meta"
    name = task_dir.name

    episodes = [json.loads(l) for l in open(meta / "episodes.jsonl")]
    vocab = [l for l in
             __import__("pandas").read_parquet(meta / "subtasks.parquet")
             .reset_index().itertuples(index=False)] if (meta / "subtasks.parquet").exists() else None
    if vocab is None:
        return [f"{name}: subtasks.parquet missing"]
    vocab_pos = {t.subtask: t.subtask_index for t in vocab}
    if sorted(vocab_pos.values()) != list(range(len(vocab))):
        errs.append(f"{name}: subtask_index not 0..N-1 contiguous")

    info = json.loads((meta / "info.json").read_text())
    if info.get("features", {}).get("subtask_index", {}).get("dtype") != "int64":
        errs.append(f"{name}: info.json missing subtask_index feature")

    # schema/layout parity with EgoStandard
    pf = pq.ParquetFile(meta / "subtasks.parquet")
    if [f.name for f in pf.schema_arrow] != ["subtask_index", "subtask"]:
        errs.append(f"{name}: subtasks.parquet schema mismatch: {pf.schema_arrow}")
    pmd = pf.schema_arrow.metadata.get(b"pandas", b"").decode()
    if '"index_columns": ["subtask"]' not in pmd:
        errs.append(f"{name}: subtasks.parquet pandas index != subtask")

    # exact per-frame comparison, episode by episode
    exp_by_ep = {ep["episode_index"]: expected_subtask_array(ep, vocab_pos)
                 for ep in episodes}
    got = {int(e): None for e in exp_by_ep}
    for df in sorted((canon / "data").glob("chunk-*/*.parquet")):
        tbl = pq.read_table(df, columns=["episode_index", "frame_index",
                                         "subtask_index"])
        ep_arr = tbl.column("episode_index").to_numpy()
        sub = tbl.column("subtask_index").to_numpy()
        if sub.dtype != np.int64:
            errs.append(f"{name}: subtask_index dtype {sub.dtype}")
        change = np.flatnonzero(np.diff(ep_arr)) + 1
        bounds = np.concatenate(([0], change, [len(ep_arr)]))
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            e = int(ep_arr[lo])
            got[e] = sub[lo:hi] if got.get(e) is None else np.concatenate([got[e], sub[lo:hi]])
    for e, exp in exp_by_ep.items():
        g = got.get(e)
        if g is None:
            errs.append(f"{name}: episode {e} missing in data parquet")
        elif len(g) != len(exp):
            errs.append(f"{name}: episode {e} length {len(g)} != {len(exp)}")
        elif not np.array_equal(g, exp):
            nbad = int(np.sum(g != exp))
            errs.append(f"{name}: episode {e} {nbad}/{len(exp)} frames mismatch")

    if official:
        # Mirror egostandard_55d_official_load.py: use ds.hf_dataset (no
        # torchcodec/torchvision video decode — both are broken in this env)
        ds = LeRobotDataset(f"local/{name}", root=str(canon))
        row = ds.hf_dataset[0]
        if "subtask_index" not in row:
            errs.append(f"{name}: official load has no subtask_index")
        elif ds.num_frames != sum(e["length"] for e in episodes):
            errs.append(f"{name}: official num_frames mismatch")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("agibot_root")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--official", action="store_true")
    args = ap.parse_args()

    root = Path(args.agibot_root)
    names = sorted(p.name for p in root.iterdir()
                   if p.is_dir() and (p / "canonical_55d" / MARKER).exists()) \
        if (MARKER := ".subtask050") else []
    if args.tasks:
        keep = set(args.tasks.split(","))
        names = [n for n in names if n in keep]
    all_errs = []
    for n in names:
        errs = validate_task(root / n, args.official)
        print(f"{n}: {'OK' if not errs else 'FAIL'}", flush=True)
        all_errs += errs
    print(f"\n{len(names)} tasks checked, {len(all_errs)} errors")
    for e in all_errs[:20]:
        print(" ", e)
    return 1 if all_errs else 0


if __name__ == "__main__":
    sys.exit(main())
