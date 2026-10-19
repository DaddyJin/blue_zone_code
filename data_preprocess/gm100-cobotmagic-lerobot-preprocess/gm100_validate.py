#!/usr/bin/env python
"""Validate a converted GM100 task dataset (v3.0 + 55D) against the spec.

Checks (per standard_format/README.md):
  1. info.json: v3.0, fps=30, required features registered (incl. subtask_index)
  2. data parquet: exact column order, dtypes, 55D width 55
  3. per-episode: frame_index 0..T-1, timestamp == frame_index/30, subtask_index == -1
  4. mask_state/mask_action/reliability: active slots == {0:6, 7:13, 14:28, 28:30};
     rest 0 (slots 6/13 and 30:55 all-zero); eef quaternions normalized
  5. action_unified[t] == state_unified[t+1] within episode; last frame copies itself
  6. frame accounting: sum(lengths) == total rows == info.total_frames
  7. episodes index: contiguous dataset_from/to_index covering [0, N); video locators sane
  8. videos: per-episode files exist for every camera; sampled PyAV duration check
  9. vocab parquets: tasks non-empty, subtasks empty with correct schema
 10. (soft) lerobot LeRobotDataset load attempt -- reported, not fatal

Usage (cosmos-framework env):
  python gm100_validate.py --task task_00001     # specific task(s)
  python gm100_validate.py --all                 # every dataset under --out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
N_DIM = 55
ACTIVE_SLOTS = set(list(range(0, 6)) + list(range(7, 13)) +
                   list(range(14, 28)) + [28, 29])
EMPTY_SLOTS = set(range(N_DIM)) - ACTIVE_SLOTS

CANON_COLS = [
    "frame_index", "episode_index", "timestamp", "task_index",
    "state_unified", "action_unified", "mask_state", "mask_action", "reliability",
    "subtask_index",
]
PASSTHROUGH_COLS = [
    "observation.state.arm.velocity",
    "observation.state.arm.effort",
    "observation.state.effector.effort",
    "action.arm.position",
    "action.effector.position",
]
DATA_COL_ORDER = CANON_COLS + PASSTHROUGH_COLS + ["index"]

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [OK]   {msg}")
    else:
        print(f"  [FAIL] {msg}")
        FAILURES.append(msg)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    WARNINGS.append(msg)


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def to_np(tbl: pa.Table, col: str) -> np.ndarray:
    return np.stack(tbl.column(col).to_pylist())


def validate_task(out_root: Path, task: str) -> None:
    d = out_root / task
    print(f"validating: {d}")
    if not d.is_dir():
        print(f"  [FAIL] task dir missing: {d}")
        FAILURES.append("task dir missing")
        return

    # ---------------- 1. info.json ----------------
    section("1. info.json")
    info = json.loads((d / "meta" / "info.json").read_text())
    check(info.get("codebase_version") == "v3.0", f"codebase_version == v3.0")
    check(info.get("fps") == FPS, f"fps == {FPS}")
    feats = info.get("features", {})
    for c in ["frame_index", "episode_index", "timestamp", "task_index",
              "state_unified", "action_unified", "mask_state", "mask_action",
              "reliability", "subtask_index", "index"]:
        check(c in feats, f"feature registered: {c}")
    check(feats.get("subtask_index", {}).get("dtype") == "int64",
          "subtask_index dtype int64")
    for c in ("state_unified", "action_unified", "mask_state", "mask_action", "reliability"):
        check(feats.get(c, {}).get("shape") == [N_DIM], f"{c} shape [55]")
    cams = sorted(k.replace("observation.images.", "") for k, v in feats.items()
                  if v.get("dtype") == "video")
    check(len(cams) == 3, f"3 video features: {cams}")
    total_frames = info["total_frames"]
    total_episodes = info["total_episodes"]

    # ---------------- 2. data parquet schema ----------------
    section("2. data parquet schema")
    data_dir = d / "data" / "chunk-000"
    files = sorted(data_dir.glob("file-*.parquet"))
    check(len(files) >= 1, f"data files found: {[f.name for f in files]}")
    schemas = {f.name: pq.ParquetFile(f).schema_arrow for f in files}
    first = schemas[files[0].name]
    check(first.names == DATA_COL_ORDER,
          f"column order matches spec ({len(first.names)} cols)")
    if first.names != DATA_COL_ORDER:
        print(f"      got: {first.names}")
    for c in CANON_COLS + PASSTHROUGH_COLS + ["index"]:
        if c not in first.names:
            continue
        t = first.field(c).type
        if c in ("state_unified", "action_unified", "mask_state", "mask_action", "reliability"):
            check(pa.types.is_fixed_size_list(t) and t.list_size == N_DIM,
                  f"{c}: fixed_size_list<float32>[55]")
        elif c == "timestamp":
            check(pa.types.is_float64(t), f"{c}: float64")
        elif c in ("frame_index", "episode_index", "task_index", "subtask_index", "index"):
            check(pa.types.is_int64(t), f"{c}: int64")
        else:
            check(pa.types.is_fixed_size_list(t), f"{c}: fixed_size_list[{t.list_size}]")
    for f in files[1:]:
        check(schemas[f.name].names == first.names, f"{f.name}: same column order")

    # ---------------- 3-6. load + per-episode checks ----------------
    section("3-6. per-frame / per-episode integrity")
    ep_meta = [json.loads(l) for l in (d / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    ep_meta.sort(key=lambda r: r["episode_index"])
    lengths = {e["episode_index"]: e["length"] for e in ep_meta}
    sum_len = sum(lengths.values())
    check(sum_len == total_frames, f"sum(episodes.length)={sum_len} == info.total_frames={total_frames}")
    check(len(ep_meta) == total_episodes, f"episodes.jsonl rows={len(ep_meta)} == total_episodes")

    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    check(total_rows == total_frames, f"data rows={total_rows} == total_frames={total_frames}")

    mask_bad = 0
    eef_bad = 0
    quat_bad = 0
    tail_bad = 0
    action_bad = 0
    frame_bad = 0
    ts_bad = 0
    subtask_bad = 0
    idx_bad = 0
    expected_global = 0
    eef_min = np.full(6, np.inf)
    eef_max = np.full(6, -np.inf)

    for f in files:
        tbl = pq.read_table(f)
        state = to_np(tbl, "state_unified")
        action = to_np(tbl, "action_unified")
        ms = to_np(tbl, "mask_state")
        ma = to_np(tbl, "mask_action")
        rel = to_np(tbl, "reliability")
        epis = tbl.column("episode_index").to_numpy()
        findex = tbl.column("frame_index").to_numpy()
        ts = tbl.column("timestamp").to_numpy()
        sti = tbl.column("subtask_index").to_numpy()
        gidx = tbl.column("index").to_numpy()

        # index contiguous 0..N-1
        if gidx[0] != expected_global or gidx[-1] != expected_global + len(gidx) - 1:
            idx_bad += 1
        expected_global += len(gidx)

        # subtask all -1
        if not np.all(sti == -1):
            subtask_bad += 1

        # split per episode (rows of one episode are contiguous within a file)
        bounds = np.flatnonzero(np.diff(epis)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(epis)]])
        for s, e in zip(starts, ends):
            ep = int(epis[s])
            T = e - s
            if T != lengths.get(ep):
                frame_bad += 1
            if not np.array_equal(findex[s:e], np.arange(T)):
                frame_bad += 1
            if not np.allclose(ts[s:e], np.arange(T) / FPS, atol=1e-6):
                ts_bad += 1
            # mask pattern
            mrow = ms[s]
            active = set(np.flatnonzero(mrow > 0.5).tolist())
            if active != ACTIVE_SLOTS:
                mask_bad += 1
            if not np.all(ms[s:e] == mrow) or not np.all(ma[s:e] == mrow) or not np.all(rel[s:e] == mrow):
                mask_bad += 1
            # empty slots (6, 13, 30:55) all zero
            if not np.all(state[s:e][:, sorted(EMPTY_SLOTS)] == 0.0):
                eef_bad += 1
            if not np.all(action[s:e][:, sorted(EMPTY_SLOTS)] == 0.0):
                eef_bad += 1
            # eef quaternions normalized (slots 17:21 = left, 24:28 = right)
            for a, b in ((17, 21), (24, 28)):
                qn = np.linalg.norm(state[s:e, a:b], axis=1)
                if not np.all(np.abs(qn - 1.0) < 1e-3):
                    quat_bad += 1
            # eef position stats (informational)
            pl = state[s:e, 14:17]
            pr = state[s:e, 21:24]
            eef_min[:3] = np.minimum(eef_min[:3], pl.min(axis=0))
            eef_max[:3] = np.maximum(eef_max[:3], pl.max(axis=0))
            eef_min[3:] = np.minimum(eef_min[3:], pr.min(axis=0))
            eef_max[3:] = np.maximum(eef_max[3:], pr.max(axis=0))
            # action[t] == state[t+1]; last copies itself
            if T > 1 and not np.array_equal(action[s:e - 1], state[s + 1:e]):
                action_bad += 1
            if not np.array_equal(action[e - 1], state[e - 1]):
                action_bad += 1

    check(mask_bad == 0, f"mask/reliability pattern (active={sorted(ACTIVE_SLOTS)}, rest 0) on every episode")
    check(eef_bad == 0, "empty slots (6, 13, hand/waist/head/mobile/reserved 30:55) all zero")
    check(quat_bad == 0, "eef quaternions normalized (slots 17:21, 24:28)")
    print(f"  [INFO] eef position range (m): left  x[{eef_min[0]:.3f},{eef_max[0]:.3f}] "
          f"y[{eef_min[1]:.3f},{eef_max[1]:.3f}] z[{eef_min[2]:.3f},{eef_max[2]:.3f}]")
    print(f"  [INFO] eef position range (m): right x[{eef_min[3]:.3f},{eef_max[3]:.3f}] "
          f"y[{eef_min[4]:.3f},{eef_max[4]:.3f}] z[{eef_min[5]:.3f},{eef_max[5]:.3f}]")
    check(action_bad == 0, "action_unified[t] == state_unified[t+1]; last frame copies itself")
    check(frame_bad == 0, "frame_index 0..T-1 per episode, lengths match episodes.jsonl")
    check(ts_bad == 0, "timestamp == frame_index/30")
    check(subtask_bad == 0, "subtask_index == -1 everywhere (no subtask labels)")
    check(idx_bad == 0, "global index contiguous 0..N-1 across files")

    # ---------------- 7. episodes index ----------------
    section("7. episodes index (meta/episodes)")
    ep_idx_files = sorted((d / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    check(len(ep_idx_files) == 1, f"episodes index file(s): {[f.name for f in ep_idx_files]}")
    ei = pq.read_table(ep_idx_files[0])
    check(ei.num_rows == total_episodes, f"episodes index rows={ei.num_rows} == {total_episodes}")
    frm = ei.column("dataset_from_index").to_numpy()
    to = ei.column("dataset_to_index").to_numpy()
    check(frm[0] == 0 and to[-1] == total_frames, "index covers [0, total_frames)")
    check(np.array_equal(frm[1:], to[:-1]), "dataset_from/to_index contiguous")
    check(np.array_equal(to - frm, ei.column("length").to_numpy()), "index lengths == episodes.jsonl lengths")
    for cam in cams:
        key = f"videos/observation.images.{cam}"
        fidx = ei.column(f"{key}/file_index").to_numpy()
        check(np.array_equal(fidx, ei.column("episode_index").to_numpy()),
              f"{cam}: video file_index == episode_index")

    # ---------------- 8. videos ----------------
    section("8. videos")
    for cam in cams:
        vdir = d / "videos" / f"observation.images.{cam}" / "chunk-000"
        n = len(list(vdir.glob("file-*.mp4")))
        check(n == total_episodes, f"{cam}: {n} video files == {total_episodes} episodes")
    # sampled duration check via PyAV
    try:
        import av
        sample_eps = sorted(lengths)[:2] + sorted(lengths)[-1:]
        for cam in cams:
            for ep in sample_eps:
                p = d / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep:03d}.mp4"
                with av.open(str(p)) as c:
                    dur = c.duration / av.time_base if c.duration else None
                exp = lengths[ep] / FPS
                if dur is None:
                    warn(f"{cam} ep{ep}: no container duration")
                elif abs(dur - exp) > 0.15:
                    warn(f"{cam} ep{ep}: duration {dur:.3f}s vs expected {exp:.3f}s")
                else:
                    print(f"  [OK]   {cam} ep{ep}: duration {dur:.3f}s ~ expected {exp:.3f}s")
    except ImportError:
        warn("PyAV not available: skipped video duration sampling")

    # ---------------- 9. vocab parquets ----------------
    section("9. vocab parquets")
    tasks_t = pq.read_table(d / "meta" / "tasks.parquet")
    check(tasks_t.column_names == ["task_index", "task"], "tasks.parquet columns")
    check(tasks_t.num_rows >= 1, f"tasks.parquet rows={tasks_t.num_rows}")
    st_t = pq.read_table(d / "meta" / "subtasks.parquet")
    check(st_t.column_names == ["subtask_index", "subtask"], "subtasks.parquet columns")
    check(st_t.num_rows == 0, "subtasks.parquet empty (no subtask labels)")

    # ---------------- 10. lerobot load (soft) ----------------
    section("10. lerobot LeRobotDataset load (soft check)")
    try:
        import lerobot
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        try:
            ds = LeRobotDataset(f"local/{task}", root=str(d))
            n = len(ds)
            check(n == total_frames, f"LeRobotDataset len={n} == total_frames")
        except Exception as e:  # noqa: BLE001
            warn(f"LeRobotDataset load failed under lerobot {lerobot.__version__}: "
                 f"{type(e).__name__}: {e}")
    except ImportError:
        warn("lerobot not importable: skipped official load")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data5T/liujin/code/data_preprocess/gm100-cobotmagic-lerobot-preprocess/processed_dataset")
    ap.add_argument("--task", nargs="+", default=None,
                    help="task dir name(s); if omitted, all datasets under --out are validated")
    ap.add_argument("--all", action="store_true",
                    help="validate every dataset dir under --out (large batches)")
    args = ap.parse_args()
    out_root = Path(args.out)

    if args.all or not args.task:
        tasks = sorted(d.name for d in out_root.iterdir()
                       if d.is_dir() and (d / "meta" / "info.json").exists())
    else:
        tasks = args.task
    if not tasks:
        print(f"no task datasets found under {out_root}")
        sys.exit(1)
    print(f"validating {len(tasks)} task(s) under {out_root}")

    failed_tasks: list[str] = []
    for t in tasks:
        n0 = len(FAILURES)
        validate_task(out_root, t)
        if len(FAILURES) > n0:
            failed_tasks.append(t)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: FAILED ({len(FAILURES)} failures, {len(WARNINGS)} warnings, "
              f"{len(failed_tasks)}/{len(tasks)} tasks failed)")
        for m in FAILURES:
            print(f"  - {m}")
        if failed_tasks:
            print("FAILED_TASKS: " + " ".join(failed_tasks))
        sys.exit(1)
    print(f"RESULT: ALL CHECKS PASSED ({len(WARNINGS)} warnings, {len(tasks)} tasks)")
    for m in WARNINGS:
        print(f"  - {m}")


if __name__ == "__main__":
    main()
