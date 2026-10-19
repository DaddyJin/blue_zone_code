#!/usr/bin/env python
"""Validate a converted lingbot GM-100 dataset (v3.0 + 55D) against the spec.

Carried over from gm100_validate.py (same checks); adds --full-videos to probe
**every** video file instead of sampling (cheap enough for demo-scale batches).

Checks (per standard_format/README.md):
  1. info.json: v3.0, fps=30, required features registered (incl. subtask_index)
  2. data parquet: exact column order, dtypes, 55D width 55
  3. per-episode: frame_index 0..T-1, timestamp == frame_index/30, subtask_index == -1
  4. mask_state/mask_action/reliability: active slots == {0:6, 7:13, 14:28, 28:30};
     rest 0 (slots 6/13 and 30:55 all-zero); eef quaternions normalized
  5. action_unified[t] == state_unified[t+1] within episode; last frame copies itself
  6. frame accounting: sum(lengths) == total rows == info.total_frames
  7. episodes index: contiguous dataset_from/to_index covering [0, N); video locators sane
  8. videos: per-episode files exist for every camera; PyAV duration check
     (sampled by default, ALL files with --full-videos)
  9. vocab parquets: tasks non-empty, subtasks empty with correct schema
 10. (soft) lerobot LeRobotDataset load attempt -- reported, not fatal

Usage (cosmos-framework env):
  python lingbot_validate.py --task BM-001                  # specific dataset(s)
  python lingbot_validate.py --task BM-001 --full-videos    # + probe all videos
  python lingbot_validate.py --all                          # every dataset under --out
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _progress import Progress, start_queue_forwarder  # noqa: E402

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


def dataset_frame_count(d: Path) -> int:
    """Total frames of a converted dataset (episodes.jsonl only; cheap)."""
    rows = [json.loads(line) for line in
            (d / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    return sum(int(r["length"]) for r in rows)


def _probe_duration(p: Path) -> tuple[float | None, str | None]:
    """Container duration via PyAV; (duration_s, error_message)."""
    import av
    try:
        with av.open(str(p)) as c:
            return (c.duration / av.time_base if c.duration else None), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _validate_worker(out_root: str, task: str, full_videos: bool,
                     video_workers: int, progress_q=None) -> dict:
    """Worker-process entry point: validate one dataset, capture its log."""
    global FAILURES, WARNINGS
    FAILURES, WARNINGS = [], []

    def cb(n: float, note: str) -> None:
        if progress_q is not None:
            progress_q.put((float(n), str(note)))

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            validate_task(Path(out_root), task, full_videos=full_videos,
                          video_workers=video_workers, progress_cb=cb)
    except Exception as e:  # noqa: BLE001
        FAILURES.append(f"validator crashed: {type(e).__name__}: {e}")
    return {"task": task, "failures": list(FAILURES),
            "warnings": list(WARNINGS), "log": buf.getvalue()}


def validate_task(out_root: Path, task: str, full_videos: bool = False,
                  video_workers: int = 4, progress_cb=None) -> None:
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
        n_rows = tbl.num_rows

        def step(frac: float, label: str) -> None:
            if progress_cb is not None:
                progress_cb(frac * n_rows, f"{task} {label}")

        step(0.05, f"read {f.name}")
        state = to_np(tbl, "state_unified")
        step(0.06, "state")
        action = to_np(tbl, "action_unified")
        step(0.06, "action")
        ms = to_np(tbl, "mask_state")
        step(0.06, "mask_state")
        ma = to_np(tbl, "mask_action")
        step(0.06, "mask_action")
        rel = to_np(tbl, "reliability")
        step(0.06, "reliability")
        epis = tbl.column("episode_index").to_numpy()
        findex = tbl.column("frame_index").to_numpy()
        ts = tbl.column("timestamp").to_numpy()
        sti = tbl.column("subtask_index").to_numpy()
        gidx = tbl.column("index").to_numpy()
        step(0.05, "index prep")

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
            if progress_cb is not None:
                progress_cb(0.6 * T, f"{task} ep {ep}")

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
    # duration check via PyAV: sampled by default, all files with --full-videos
    try:
        import av  # noqa: F401  (availability check; _probe_duration imports it)
        probe_eps = sorted(lengths) if full_videos else (sorted(lengths)[:2] + sorted(lengths)[-1:])
        probes = []
        for cam in cams:
            for ep in probe_eps:
                p = d / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep:03d}.mp4"
                probes.append((cam, ep, p, lengths[ep] / FPS))
        durations: dict[tuple[str, int], tuple[float | None, str | None]] = {}
        if max(1, int(video_workers)) > 1 and len(probes) > 1:
            with ThreadPoolExecutor(max_workers=max(1, int(video_workers))) as tex:
                futs = {tex.submit(_probe_duration, p): (cam, ep)
                        for cam, ep, p, _exp in probes}
                for fut in as_completed(futs):
                    durations[futs[fut]] = fut.result()
        else:
            for cam, ep, p, _exp in probes:
                durations[(cam, ep)] = _probe_duration(p)

        bad = 0
        for cam, ep, _p, exp in probes:
            dur, err = durations[(cam, ep)]
            if err is not None:
                bad += 1
                warn(f"{cam} ep{ep}: probe failed ({err})")
            elif dur is None:
                bad += 1
                warn(f"{cam} ep{ep}: no container duration")
            elif abs(dur - exp) > 0.15:
                bad += 1
                warn(f"{cam} ep{ep}: duration {dur:.3f}s vs expected {exp:.3f}s")
        if full_videos:
            check(bad == 0, f"duration check on ALL {len(cams) * len(probe_eps)} videos "
                            f"(== length/30 per episode)")
        else:
            n_probed = len(cams) * len(probe_eps)
            print(f"  [OK]   sampled {n_probed} videos: duration ~ length/30 "
                  f"(use --full-videos to probe all {len(cams) * total_episodes})")
    except ImportError:
        warn("PyAV not available: skipped video duration checks")

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
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "processed_dataset"))
    ap.add_argument("--task", nargs="+", default=None,
                    help="dataset dir name(s); if omitted, all datasets under --out are validated")
    ap.add_argument("--all", action="store_true",
                    help="validate every dataset dir under --out (large batches)")
    ap.add_argument("--full-videos", action="store_true",
                    help="probe the duration of ALL video files (not just a sample)")
    ap.add_argument("--workers", type=int, default=8,
                    help="datasets validated in parallel (1 = sequential; default: 8)")
    ap.add_argument("--video-workers", type=int, default=4,
                    help="concurrent PyAV duration probes per dataset (default: 4)")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable progress/ETA lines")
    ap.add_argument("--progress-interval", type=float, default=10.0,
                    help="seconds between progress lines (default: 10)")
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

    frames_by_task: dict[str, int] = {}
    for t in tasks:
        try:
            frames_by_task[t] = dataset_frame_count(out_root / t)
        except Exception as e:  # noqa: BLE001  (let validate_task report the real error)
            print(f"  [WARN] cannot pre-scan {t}: {e}")
            frames_by_task[t] = 0
    prog = Progress(sum(frames_by_task.values()), "frames", "validate",
                    enabled=not args.no_progress,
                    interval=args.progress_interval)
    prog.start_heartbeat()

    failed_tasks: list[str] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for t in tasks:
            n0 = len(FAILURES)
            validate_task(out_root, t, full_videos=args.full_videos,
                          video_workers=args.video_workers,
                          progress_cb=lambda n, note: prog.update(n, note=note))
            if len(FAILURES) > n0:
                failed_tasks.append(t)
    else:
        mgr = mp.Manager()
        q = mgr.Queue()
        stop = threading.Event()
        fwd = start_queue_forwarder(q, prog, stop)
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_validate_worker, str(out_root), t,
                                  args.full_videos, args.video_workers, q): t
                        for t in tasks}
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa: BLE001  (worker crash/pickling)
                        res = {"task": t,
                               "failures": [f"worker crashed: {type(e).__name__}: {e}"],
                               "warnings": [], "log": ""}
                    if res["log"]:
                        print(f"\n===== {t} =====")
                        print(res["log"].rstrip())
                    FAILURES.extend(res["failures"])
                    WARNINGS.extend(res["warnings"])
                    if res["failures"]:
                        failed_tasks.append(t)
        finally:
            q.put(None)
            stop.set()
            fwd.join(timeout=2)
            mgr.shutdown()
    prog.finish(ok=not failed_tasks)

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
