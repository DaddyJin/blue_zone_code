#!/usr/bin/env python
"""Validate a converted InternData-A1 dataset (v3.0 + 55D) against the spec.

Checks (per standard_format/README.md, incl. the §7 validation chain):

 1. info.json: v3.0, fps=30, required features (incl. subtask_index int64),
    data_files_size_in_mb=100 / video_files_size_in_mb=500 (spec §4.1)
 2. data parquet: exact column order (33 cols), dtypes, 55D width 55
 3. per-episode: frame_index 0..T-1, timestamp == frame_index/30, subtask_index == -1
 4. mask_state/mask_action/reliability: active slots == {0:7, 14:21, 28}; rest 0;
    eef quaternions normalized
 5. action_unified[t] == state_unified[t+1] within episode; last frame copies itself
 6. frame accounting: sum(lengths) == total rows == info.total_frames
 7. FK cross-check: standard Franka DH forward kinematics (intern_fk, joints only)
    vs the 55D eef values (from source gripper.pose) -- independent verification
 8. episodes index: contiguous dataset_from/to_index covering [0, N); video locators:
    per-file cumulative from/to_timestamp chain (each file starts at 0)
 9. videos: per-camera file counts, ALL <=960s (spec §7 durcap), full frame-count
    check per file, drift |sum durations - sum frames/30| <= 0.5s (spec §7 #1),
    moov-at-front (+faststart), first-frame decode per file, pts uniformity (stts)
10. vocab parquets; subtasks empty with correct schema
11. (soft) lerobot LeRobotDataset load + a few random __getitem__ decode calls

Usage (cosmos-framework env):
  python intern_validate.py --task close_the_laptop
  python intern_validate.py --all
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intern_fk import fk_flange  # noqa: E402

FPS = 30
N_DIM = 55
CAP_S = 960.0
ACTIVE_SLOTS = set(list(range(0, 7)) + list(range(14, 21)) + [28])
EMPTY_SLOTS = set(range(N_DIM)) - ACTIVE_SLOTS

CANON_COLS = [
    "frame_index", "episode_index", "timestamp", "task_index",
    "state_unified", "action_unified", "mask_state", "mask_action", "reliability",
    "subtask_index",
]
PASSTHROUGH_COLS = [
    "head_camera_intrinsics", "hand_camera_intrinsics",
    "head_camera_to_robot_extrinsics", "hand_camera_to_robot_extrinsics",
    "states.gripper.pose",
    "states.ee_to_armbase_pose", "states.ee_to_robot_pose",
    "states.tcp_to_armbase_pose", "states.tcp_to_robot_pose",
    "states.robot_to_env_pose",
    "actions.joint.position", "actions.gripper.position", "actions.gripper.openness",
    "actions.gripper.pose",
    "actions.ee_to_armbase_pose", "actions.ee_to_robot_pose",
    "actions.tcp_to_armbase_pose", "actions.tcp_to_robot_pose",
    "master_actions.joint.position", "master_actions.gripper.position",
    "master_actions.gripper.openness", "master_actions.gripper.pose",
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


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(q1 * q2, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


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
    check(info.get("codebase_version") == "v3.0", "codebase_version == v3.0")
    check(info.get("fps") == FPS, f"fps == {FPS}")
    check(info.get("data_files_size_in_mb") == 100, "data_files_size_in_mb == 100")
    check(info.get("video_files_size_in_mb") == 500, "video_files_size_in_mb == 500")
    feats = info.get("features", {})
    for c in CANON_COLS + ["index"]:
        check(c in feats, f"feature registered: {c}")
    check(feats.get("subtask_index", {}).get("dtype") == "int64",
          "subtask_index dtype int64")
    for c in ("state_unified", "action_unified", "mask_state", "mask_action", "reliability"):
        check(feats.get(c, {}).get("shape") == [N_DIM], f"{c} shape [55]")
    cams = sorted(k.replace("observation.images.", "") for k, v in feats.items()
                  if v.get("dtype") == "video")
    check(len(cams) == 2 and cams == ["hand", "head"], f"2 video features: {cams}")
    for cam in cams:
        v = feats[f"observation.images.{cam}"]
        check(v.get("info", {}).get("video.codec") == "h264",
              f"{cam}: codec h264 (spec §5)")
        check(v.get("info", {}).get("video.fps") == FPS, f"{cam}: video fps == 30")
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
    for c in DATA_COL_ORDER:
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
            check(pa.types.is_fixed_size_list(t) or pa.types.is_float32(t),
                  f"{c}: {t}")
    for f in files[1:]:
        check(schemas[f.name].names == first.names, f"{f.name}: same column order")

    # ---------------- 3-7. per-frame / per-episode integrity ----------------
    section("3-7. per-frame / per-episode integrity + FK cross-check")
    ep_meta = [json.loads(l) for l in (d / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    ep_meta.sort(key=lambda r: r["episode_index"])
    lengths = {e["episode_index"]: e["length"] for e in ep_meta}
    sum_len = sum(lengths.values())
    check(sum_len == total_frames, f"sum(episodes.length)={sum_len} == info.total_frames={total_frames}")
    check(len(ep_meta) == total_episodes, f"episodes.jsonl rows={len(ep_meta)} == total_episodes")

    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    check(total_rows == total_frames, f"data rows={total_rows} == total_frames={total_frames}")

    mask_bad = eef_bad = quat_bad = action_bad = frame_bad = ts_bad = 0
    subtask_bad = idx_bad = 0
    expected_global = 0
    fk_pos_max = 0.0
    fk_ang_max = 0.0
    eef_min = np.full(3, np.inf)
    eef_max = np.full(3, -np.inf)

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

        if gidx[0] != expected_global or gidx[-1] != expected_global + len(gidx) - 1:
            idx_bad += 1
        expected_global += len(gidx)
        if not np.all(sti == -1):
            subtask_bad += 1

        # ---- FK cross-check on the whole file (vectorized) ----
        joints = state[:, 0:7].astype(np.float64)
        pos_fk, quat_fk = fk_flange(joints)
        pos_err = np.abs(pos_fk - state[:, 14:17]).max(axis=1)
        ang_err = quat_angle_deg(quat_fk, state[:, 17:21])
        fk_pos_max = max(fk_pos_max, float(pos_err.max()))
        fk_ang_max = max(fk_ang_max, float(ang_err.max()))

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
            mrow = ms[s]
            active = set(np.flatnonzero(mrow > 0.5).tolist())
            if active != ACTIVE_SLOTS:
                mask_bad += 1
            if not np.all(ms[s:e] == mrow) or not np.all(ma[s:e] == mrow) or not np.all(rel[s:e] == mrow):
                mask_bad += 1
            if not np.all(state[s:e][:, sorted(EMPTY_SLOTS)] == 0.0):
                eef_bad += 1
            if not np.all(action[s:e][:, sorted(EMPTY_SLOTS)] == 0.0):
                eef_bad += 1
            qn = np.linalg.norm(state[s:e, 17:21], axis=1)
            if not np.all(np.abs(qn - 1.0) < 1e-3):
                quat_bad += 1
            pl = state[s:e, 14:17]
            eef_min = np.minimum(eef_min, pl.min(axis=0))
            eef_max = np.maximum(eef_max, pl.max(axis=0))
            if T > 1 and not np.array_equal(action[s:e - 1], state[s + 1:e]):
                action_bad += 1
            if not np.array_equal(action[e - 1], state[e - 1]):
                action_bad += 1

    check(mask_bad == 0, f"mask/reliability pattern (active={sorted(ACTIVE_SLOTS)}, rest 0)")
    check(eef_bad == 0, "empty slots (7:14, 21:28, 29:55) all zero")
    check(quat_bad == 0, "eef quaternions normalized (slots 17:21)")
    check(fk_ang_max < 0.1 and fk_pos_max < 2e-4,
          f"FK(joints) == 55D eef (gripper.pose): max pos {fk_pos_max:.2e} m, "
          f"max angle {fk_ang_max:.4f} deg")
    print(f"  [INFO] eef position range (m): x[{eef_min[0]:.3f},{eef_max[0]:.3f}] "
          f"y[{eef_min[1]:.3f},{eef_max[1]:.3f}] z[{eef_min[2]:.3f},{eef_max[2]:.3f}]")
    check(action_bad == 0, "action_unified[t] == state_unified[t+1]; last frame copies itself")
    check(frame_bad == 0, "frame_index 0..T-1 per episode, lengths match episodes.jsonl")
    check(ts_bad == 0, "timestamp == frame_index/30")
    check(subtask_bad == 0, "subtask_index == -1 everywhere (no subtask labels)")
    check(idx_bad == 0, "global index contiguous 0..N-1 across files")

    # ---------------- 8. episodes index ----------------
    section("8. episodes index (meta/episodes) + video locator chain")
    ep_idx_files = sorted((d / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    check(len(ep_idx_files) == 1, f"episodes index file(s): {[f.name for f in ep_idx_files]}")
    ei = pq.read_table(ep_idx_files[0])
    check(ei.num_rows == total_episodes, f"episodes index rows={ei.num_rows} == {total_episodes}")
    frm = ei.column("dataset_from_index").to_numpy()
    to = ei.column("dataset_to_index").to_numpy()
    check(frm[0] == 0 and to[-1] == total_frames, "index covers [0, total_frames)")
    check(np.array_equal(frm[1:], to[:-1]), "dataset_from/to_index contiguous")
    check(np.array_equal(to - frm, ei.column("length").to_numpy()),
          "index lengths == episodes.jsonl lengths")

    cam_file_eps: dict[str, dict[int, list[int]]] = {}
    for cam in cams:
        key = f"videos/observation.images.{cam}"
        fidx = ei.column(f"{key}/file_index").to_numpy()
        fts = ei.column(f"{key}/from_timestamp").to_numpy()
        tts = ei.column(f"{key}/to_timestamp").to_numpy()
        eps_l = ei.column("episode_index").to_numpy()
        bad_chain = 0
        by_file: dict[int, list[int]] = {}
        for ep, fi, a, b in zip(eps_l, fidx, fts, tts):
            L = ei.column("length").to_numpy()[ep]
            if abs((b - a) - L / FPS) > 1e-6:
                bad_chain += 1
            by_file.setdefault(int(fi), []).append(int(ep))
        for fi, epl in by_file.items():
            a0 = fts[epl[0]]
            if abs(a0) > 1e-6:
                bad_chain += 1
            cum = 0.0
            for ep in epl:
                if abs(fts[ep] - cum) > 1e-6:
                    bad_chain += 1
                cum += ei.column("length").to_numpy()[ep] / FPS
                if abs(tts[ep] - cum) > 1e-6:
                    bad_chain += 1
        check(bad_chain == 0, f"{cam}: from/to_timestamp per-file cumulative chain "
                              f"({len(by_file)} files)")
        cam_file_eps[cam] = by_file

    # ---------------- 9. videos ----------------
    section("9. videos (durcap / drift / frame counts)")
    try:
        import av
    except ImportError:
        warn("PyAV not available: skipped video checks")
        av = None

    if av is not None:
        for cam in cams:
            key = f"observation.images.{cam}"
            vdir = d / "videos" / key / "chunk-000"
            n_files = len(list(vdir.glob("file-*.mp4")))
            n_expected = len(cam_file_eps[cam])
            check(n_files == n_expected, f"{cam}: {n_files} video files == {n_expected} expected")
            exp_hw = tuple(feats[key]["shape"][:2])
            dur_sum = 0.0
            exp_sum = 0.0
            problems = 0
            for fi, epl in sorted(cam_file_eps[cam].items()):
                p = vdir / f"file-{fi:03d}.mp4"
                exp_frames = int(sum(ei.column("length").to_numpy()[ep] for ep in epl))
                exp_dur = exp_frames / FPS
                exp_sum += exp_dur
                if not p.exists():
                    problems += 1
                    warn(f"{cam} file-{fi:03d}: missing")
                    continue
                with av.open(str(p)) as c:
                    s = c.streams.video[0]
                    dur = c.duration / av.time_base if c.duration else None
                    nf = s.frames
                    w, h = s.codec_context.width, s.codec_context.height
                    frame = next(c.decode(s), None)   # real decode (first frame)
                if dur is None:
                    problems += 1
                    warn(f"{cam} file-{fi:03d}: no container duration")
                else:
                    dur_sum += dur
                    if dur > CAP_S + 1.0:
                        problems += 1
                        warn(f"{cam} file-{fi:03d}: duration {dur:.2f}s > {CAP_S}s cap")
                if nf and nf != exp_frames:
                    problems += 1
                    warn(f"{cam} file-{fi:03d}: {nf} frames != expected {exp_frames}")
                if (w, h) != (exp_hw[1], exp_hw[0]):
                    problems += 1
                    warn(f"{cam} file-{fi:03d}: {w}x{h} != feature shape {exp_hw[1]}x{exp_hw[0]}")
                if frame is None:
                    problems += 1
                    warn(f"{cam} file-{fi:03d}: first frame did not decode")
                with open(p, "rb") as fh:
                    if b"moov" not in fh.read(65536):
                        problems += 1
                        warn(f"{cam} file-{fi:03d}: moov not at front (+faststart)")
            check(problems == 0, f"{cam}: {len(cam_file_eps[cam])} files verified "
                                 f"(<=960s, frame counts, decode, faststart)")
            drift = abs(dur_sum - exp_sum)
            check(drift <= 0.5, f"{cam}: drift |sum durations - sum frames/30| = "
                                f"{drift:.3f}s <= 0.5s (spec §7 #1)")

        # container timestamp uniformity (stts health analog, spec §7 #5): the
        # DECODE timestamps must form a uniform grid (this mirrors what the mp4
        # stts table encodes). Presentation (pts) order is NOT used -- B-frame
        # reordering makes pts non-monotonic in decode order, which is normal.
        for cam in cams:
            vdir = d / "videos" / f"observation.images.{cam}" / "chunk-000"
            p = sorted(vdir.glob("file-*.mp4"))[0]
            with av.open(str(p)) as c:
                s = c.streams.video[0]
                ts = [pkt.dts if pkt.dts is not None else pkt.pts
                      for pkt in c.demux(s) if pkt.dts is not None or pkt.pts is not None]
            deltas = np.diff([t for t in ts if t is not None])
            if len(deltas):
                vals, counts = np.unique(deltas, return_counts=True)
                frac = 1.0 - counts.max() / len(deltas)
                if frac > 0.02:
                    warn(f"{cam} {p.name}: dts anomaly fraction {frac:.3%} > 2%")
                else:
                    print(f"  [OK]   {cam} {p.name}: decode timestamps uniform "
                          f"(anomaly {frac:.3%} <= 2%)")

    # ---------------- 10. vocab parquets ----------------
    section("10. vocab parquets")
    tasks_t = pq.read_table(d / "meta" / "tasks.parquet")
    check(tasks_t.column_names == ["task_index", "task"], "tasks.parquet columns")
    check(tasks_t.num_rows >= 1, f"tasks.parquet rows={tasks_t.num_rows}")
    st_t = pq.read_table(d / "meta" / "subtasks.parquet")
    check(st_t.column_names == ["subtask_index", "subtask"], "subtasks.parquet columns")
    check(st_t.num_rows == 0, "subtasks.parquet empty (no subtask labels)")

    # ---------------- 11. lerobot load (soft) ----------------
    section("11. lerobot LeRobotDataset load (soft check)")
    try:
        import lerobot
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        try:
            ds = LeRobotDataset(f"local/{task}", root=str(d))
            n = len(ds)
            check(n == total_frames, f"LeRobotDataset len={n} == total_frames")
            random.seed(0)
            sample = sorted(set([0, n - 1, n // 2, 28799, 28800]
                                + random.sample(range(n), 5)))

            def _sample_ok(ds_obj):
                for i in sample:
                    ds_obj[i]          # full read incl. real video decode

            try:
                _sample_ok(ds)
                print(f"  [OK]   random __getitem__ full decode "
                      f"({len(sample)} frames incl. file boundaries)")
            except IndexError as e:
                if "positional indexer" not in str(e):
                    raise
                # Known lerobot <= 0.4.x quirk (NOT a data defect): __getitem__
                # has no guard for subtask_index == -1 with an empty subtasks
                # table (`self.meta.subtasks.iloc[-1]` on 0 rows -> IndexError).
                # The team standard env (lerobot 0.5.0) handles -1 per spec §7#4
                # ("统计 -1 占比"). Workaround for local 0.4.x: skip the lookup.
                warn(f"lerobot {lerobot.__version__} __getitem__ lacks the "
                     f"subtask_index==-1 guard (empty subtasks table); retried with "
                     f"the documented workaround ds.meta.subtasks = None")
                ds.meta.subtasks = None
                _sample_ok(ds)
                print(f"  [OK]   random __getitem__ full decode with workaround "
                      f"({len(sample)} frames incl. file boundaries)")
        except Exception as e:  # noqa: BLE001
            warn(f"LeRobotDataset load failed under lerobot {lerobot.__version__}: "
                 f"{type(e).__name__}: {e}")
    except ImportError:
        warn("lerobot not importable: skipped official load")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="/data5T/liujin/code/data_preprocess/InternData-A1-preprocess/processed_dataset")
    ap.add_argument("--task", nargs="+", default=None,
                    help="dataset dir name(s); if omitted, all datasets under --out are validated")
    ap.add_argument("--all", action="store_true",
                    help="validate every dataset dir under --out")
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
