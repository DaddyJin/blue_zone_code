#!/usr/bin/env python
"""InternData-A1 (Franka sim) lerobot v3.0 -> team unified v3.0 + 55D.

Converts the source (already LeRobot v3.0 layout, but not in team standard) into
the team's unified format (see standard_format/README.md). Differences handled:

  1. 55D columns + subtask_index + episodes.jsonl + modality.json: added/created
  2. videos: source is AV1, multi-episode packed, files up to 4672s -> transcoded
     to h264 (libx264 crf=18 preset=fast yuv420p +faststart, per spec §5) and
     re-packed at episode boundaries into <=960s files (spec §5 hard cap)
  3. camera keys renamed images.rgb.<cam> -> observation.images.<cam>

  src (v3.0)                                 dst (v3.0 + 55D)
    close_the_laptop/                          close_the_laptop/
      data/chunk-000/file-*.parquet              data/chunk-000/file-*.parquet (~100MB rotate)
      videos/images.rgb.<cam>/chunk-000/*.mp4    videos/observation.images.<cam>/chunk-000/*.mp4
      meta/info.json (v3.0)                      meta/info.json (v3.0 + 55D features)
      meta/tasks.parquet                         meta/tasks.parquet
      meta/stats.json                            meta/subtasks.parquet (empty: no labels)
      meta/episodes/chunk-000/file-*.parquet     meta/episodes.jsonl
                                                 meta/episodes/chunk-000/file-*.parquet
                                                 meta/modality.json
                                                 qc_report.json

55D slot mapping (55 = L/R arm-joint 7+7, eef 7+7, gripper 1+1, hand 6+6,
waist 4, head 2, mobile 3, reserved 4) -- single 7-DOF Franka arm -> LEFT slots:
  0:7    <- 7 joint angles (rad)             (mask 1; right arm slots empty)
  7:14   <- EMPTY (mask 0)
  14:21  <- flange eef from `states.gripper.pose`: position (m) + euler->quat
            xyzw (URDF rpy convention), robot base frame ('armbase')
  21:28  <- EMPTY (mask 0)
  28     <- left gripper raw opening (m, [0, 0.08]; NOT normalized)
  29:55  <- EMPTY (mask 0)
The source `states.ee_to_*`/`tcp_to_*` QUATERNIONS are defective (no standard
convention reproduces the true orientation) -- they are kept VERBATIM in the
passthrough columns and never used for the 55D eef slot. `gripper.pose` euler
values were verified against standard Franka FK to 0.0000 deg / 0.00000 m over
all 140,157 frames (see intern_fk.py and intern_validate.py).

action_unified[t] = state_unified[t+1] (last frame copies itself), strictly
within each episode. Source `actions.*` (command targets) and `master_actions.*`
(teleop master arm) are kept as passthrough columns.

Video packing (spec §5): episodes are greedily packed into files of <=960s
(28800 frames), splitting ONLY at episode boundaries. Transcode+segment is done
in ONE ffmpeg pass per source file. from/to_timestamp in the episodes index use
the per-file cumulative timeline (each output file starts at 0), matching the
standard samples.

Usage (cosmos-framework env):
  conda activate cosmos-framework
  python intern_extract.py                        # 1) extract archives (SKIP if done)
  python intern_convert.py --task close_the_laptop   # 2) convert
  python intern_validate.py --task close_the_laptop  # 3) validate
"""
from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intern_fk import fk_flange  # noqa: E402  (used for an in-convert sanity check)
from _progress import Progress, start_queue_forwarder, log_line  # noqa: E402

FPS = 30
DATA_ROTATE_BYTES = 100 * 1024 * 1024   # on-disk target per data file
COMPRESSION_PRIOR = 8.0                 # in-mem/disk prior before first flush
VIDEO_CAP_FRAMES = 960 * FPS            # spec §5 hard cap (s)
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# 55D slot layout (standard_format/README.md §3.2), single arm -> left slots
N_DIM = 55
LEFT_ARM_JOINT = (0, 7)
LEFT_EEF = (14, 21)      # pos(3) + quat xyzw(4)
LEFT_GRIPPER = 28
ACTIVE_SLOTS = sorted(list(range(*LEFT_ARM_JOINT)) + list(range(*LEFT_EEF)) + [LEFT_GRIPPER])

# source (v3.0) column names and their roles
SRC_JOINT = "states.joint.position"        # (7,)  -> 55D 0:7
SRC_GRIPPER = "states.gripper.position"    # scalar -> 55D slot 28
SRC_GRIPPER_POSE = "states.gripper.pose"   # (6,) pos(3)+euler(3) -> 55D 14:21

# passthrough columns (kept verbatim from source, source schema order)
PASSTHROUGH_COLS = [
    "head_camera_intrinsics",                                     # (4,)
    "hand_camera_intrinsics",                                     # (4,)
    "head_camera_to_robot_extrinsics",                            # (7,)
    "hand_camera_to_robot_extrinsics",                            # (7,)
    "states.gripper.pose",                                        # (6,)
    "states.ee_to_armbase_pose",                                  # (7,)
    "states.ee_to_robot_pose",                                    # (7,)
    "states.tcp_to_armbase_pose",                                 # (7,)
    "states.tcp_to_robot_pose",                                   # (7,)
    "states.robot_to_env_pose",                                   # (7,)
    "actions.joint.position",                                     # (7,)
    "actions.gripper.position",                                   # scalar
    "actions.gripper.openness",                                   # scalar
    "actions.gripper.pose",                                       # (6,)
    "actions.ee_to_armbase_pose",                                 # (7,)
    "actions.ee_to_robot_pose",                                   # (7,)
    "actions.tcp_to_armbase_pose",                                # (7,)
    "actions.tcp_to_robot_pose",                                  # (7,)
    "master_actions.joint.position",                              # (7,)
    "master_actions.gripper.position",                            # scalar
    "master_actions.gripper.openness",                            # scalar
    "master_actions.gripper.pose",                                # (6,)
]

# final data-column order (10 canonical + 22 passthrough + global index = 33)
CANON_COLS = [
    "frame_index", "episode_index", "timestamp", "task_index",
    "state_unified", "action_unified", "mask_state", "mask_action", "reliability",
    "subtask_index",
]
DATA_COL_ORDER = CANON_COLS + PASSTHROUGH_COLS + ["index"]

INT_COLS = {"frame_index", "episode_index", "task_index", "subtask_index", "index"}


def log(line: str) -> None:
    log_line(line)


def fixed_list_f32(arr: np.ndarray) -> pa.Array:
    """(N, D) float32 ndarray -> FixedSizeListArray<float32>[D]."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(arr.ravel(), type=pa.float32()), arr.shape[1])


def dataset_frame_count(dataset_dir: Path) -> int:
    """Total frames of one source dataset (v3.0 episodes index; cheap)."""
    files = sorted((dataset_dir / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    if files:
        return int(sum(pq.read_table(f).column("length").to_numpy().sum() for f in files))
    data_files = sorted((dataset_dir / "data" / "chunk-000").glob("file-*.parquet"))
    if data_files:
        return int(sum(pq.ParquetFile(f).metadata.num_rows for f in data_files))
    raise FileNotFoundError(f"cannot determine length for {dataset_dir}")


def dataset_work_units(dataset_dir: Path) -> int:
    """Weighted progress units: one unit per data frame + one per video frame.

    Video transcode dominates wall time, so counting video frames as well makes
    the batch-level ETA meaningful (data phase ~10%, video phase ~90%).
    """
    frames = dataset_frame_count(dataset_dir)
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    n_cams = sum(1 for v in info.get("features", {}).values()
                 if v.get("dtype") == "video")
    return frames * (1 + max(1, n_cams))


def probe_video(path: Path) -> dict:
    """ffprobe: nb_frames / width / height / container duration."""
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    st = j["streams"][0]
    nf = st.get("nb_frames")
    return {"nb_frames": int(nf) if nf not in (None, "N/A") else None,
            "width": int(st["width"]), "height": int(st["height"]),
            "duration": float(j["format"]["duration"])}


def moov_at_front(path: Path) -> bool:
    """+faststart check: 'moov' box within the first 64KB (mp4)."""
    with open(path, "rb") as f:
        head = f.read(65536)
    return b"moov" in head


def build_episode_table(src: pa.Table, ep_idx: int, global_cursor: int) -> pa.Table:
    """Build the 33-column v3.0 table for one episode.

    `src` is the source data table already sliced to this episode's rows.
    Passthrough columns are reused verbatim (zero-copy slices).
    """
    T = src.num_rows
    joints = np.stack(src.column(SRC_JOINT).to_pylist()).astype(np.float64)    # (T,7)
    grip = np.asarray(src.column(SRC_GRIPPER).to_pylist(), dtype=np.float64)   # (T,)
    gpose = np.stack(src.column(SRC_GRIPPER_POSE).to_pylist()).astype(np.float64)  # (T,6)

    # state_unified: joints + flange eef (pos + euler->quat xyzw) + gripper
    state = np.zeros((T, N_DIM), dtype=np.float64)
    state[:, LEFT_ARM_JOINT[0]:LEFT_ARM_JOINT[1]] = joints
    state[:, LEFT_EEF[0]:LEFT_EEF[0] + 3] = gpose[:, :3]
    state[:, LEFT_EEF[0] + 3:LEFT_EEF[1]] = Rotation.from_euler("xyz", gpose[:, 3:6]).as_quat()
    state[:, LEFT_GRIPPER] = grip

    # action_unified = next-frame state (last frame copies itself), in-episode only
    action = np.zeros_like(state)
    action[0:T - 1] = state[1:T]
    action[T - 1] = state[T - 1]

    # mask / reliability: which 55D slots carry real data (reliability == mask)
    mask = np.zeros((T, N_DIM), dtype=np.float64)
    mask[:, LEFT_ARM_JOINT[0]:LEFT_ARM_JOINT[1]] = 1
    mask[:, LEFT_EEF[0]:LEFT_EEF[1]] = 1
    mask[:, LEFT_GRIPPER] = 1

    cols: dict[str, pa.Array] = {
        "frame_index": pa.array(np.arange(T, dtype=np.int64)),
        "episode_index": pa.array(np.full(T, ep_idx, dtype=np.int64)),
        "timestamp": pa.array(np.arange(T, dtype=np.float64) / FPS),
        "task_index": pa.array(np.zeros(T, dtype=np.int64)),      # single-task dataset
        "state_unified": fixed_list_f32(state),
        "action_unified": fixed_list_f32(action),
        "mask_state": fixed_list_f32(mask),
        "mask_action": fixed_list_f32(mask),
        "reliability": fixed_list_f32(mask),
        "subtask_index": pa.array(np.full(T, -1, dtype=np.int64)),  # no subtask labels
    }
    for name in PASSTHROUGH_COLS:
        cols[name] = src.column(name)   # verbatim (scalar stays scalar, list stays list)
    cols["index"] = pa.array(np.arange(global_cursor, global_cursor + T, dtype=np.int64))

    return pa.table({k: cols[k] for k in DATA_COL_ORDER})


def write_data(ep_tables: list[tuple[dict, pa.Table]], data_dir: Path) -> list[dict]:
    """Concatenate episode tables, rotating into file-XXX.parquet at episode
    boundaries (~100MB on-disk, self-calibrating compression ratio).
    Returns per-episode data-location rows."""
    data_dir.mkdir(parents=True, exist_ok=True)
    pending: list[pa.Table] = []
    pending_bytes = 0
    file_index = 0
    disk_per_mem = 1.0 / COMPRESSION_PRIOR
    locs: list[dict] = []

    def flush() -> None:
        nonlocal file_index, disk_per_mem
        tbl = pa.concat_tables(pending)
        path = data_dir / f"file-{file_index:03d}.parquet"
        pq.write_table(tbl, path, compression="zstd")
        disk = path.stat().st_size
        if pending_bytes > 0:
            disk_per_mem = disk / pending_bytes
        log(f"    data file-{file_index:03d}.parquet: {tbl.num_rows} rows, "
            f"{disk / 1e6:.1f} MB on disk (in-mem {tbl.nbytes / 1e6:.1f} MB)")
        file_index += 1

    for ep, tbl in ep_tables:
        if pending and (pending_bytes + tbl.nbytes) * disk_per_mem > DATA_ROTATE_BYTES:
            flush()
            pending = []
            pending_bytes = 0
        pending.append(tbl)
        pending_bytes += tbl.nbytes
        locs.append({
            "episode_index": ep["episode_index"],
            "data/chunk_index": 0,
            "data/file_index": file_index,
            "dataset_from_index": int(tbl.column("index")[0].as_py()),
            "dataset_to_index": int(tbl.column("index")[-1].as_py()) + 1,
        })
    if pending:
        flush()
    return locs


def write_vocab_parquet(path: Path, name: str, items: list[tuple[int, str]]) -> None:
    """tasks.parquet / subtasks.parquet in the official layout
    (<name> string as pandas index + <name>_index column)."""
    col = f"{name}_index"
    tbl = pa.table({
        col: pa.array([i for i, _ in items], type=pa.int64()),
        name: pa.array([t for _, t in items], type=pa.string()),
    })
    pandas_meta = {
        "index_columns": [name],
        "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode",
                            "numpy_type": "object", "metadata": None}],
        "columns": [
            {"name": col, "field_name": col, "pandas_type": "int64",
             "numpy_type": "int64", "metadata": None},
            {"name": name, "field_name": name, "pandas_type": "unicode",
             "numpy_type": "object", "metadata": None},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    tbl = tbl.replace_schema_metadata(
        {**(tbl.schema.metadata or {}), b"pandas": json.dumps(pandas_meta).encode()})
    pq.write_table(tbl, path)


def plan_packs(ep_meta: list[dict], cam_key: str) -> list[dict]:
    """Greedy-pack episodes into <=960s output files, splitting only at episode
    boundaries. Input episodes are grouped by their SOURCE video file first
    (each source file transcodes independently in one ffmpeg pass)."""
    groups: dict[int, list[dict]] = {}
    for e in ep_meta:
        groups.setdefault(e[f"videos/{cam_key}/file_index"], []).append(e)
    packs: list[dict] = []
    for sf in sorted(groups):
        cur: list[dict] = []
        cur_frames = 0
        for e in groups[sf]:
            L = e["length"]
            if L > VIDEO_CAP_FRAMES:
                raise RuntimeError(f"episode {e['episode_index']} length {L} > cap "
                                   f"{VIDEO_CAP_FRAMES} frames ({L / FPS:.1f}s); "
                                   f"splitting inside an episode is not allowed")
            if cur and cur_frames + L > VIDEO_CAP_FRAMES:
                packs.append({"src_file": sf, "episodes": cur, "frames": cur_frames})
                cur, cur_frames = [], 0
            cur.append(e)
            cur_frames += L
        if cur:
            packs.append({"src_file": sf, "episodes": cur, "frames": cur_frames})
    return packs


def build_video_jobs(ep_meta: list[dict], cam_keys: list[str],
                     cams: list[str]) -> tuple[list[dict], dict[str, dict[int, tuple[int, float, float]]]]:
    """Plan parallel transcode jobs and the per-episode video locators.

    One job = one source video file (all of its <=960s output packs), so jobs
    never read the same source concurrently. Output file indices are fixed here,
    independent of job completion order.
    """
    jobs: list[dict] = []
    cam_locs: dict[str, dict[int, tuple[int, float, float]]] = {}
    for k, cam in zip(cam_keys, cams):
        packs = plan_packs(ep_meta, k)
        m: dict[int, tuple[int, float, float]] = {}
        for fi, pack in enumerate(packs):
            cum = 0
            for e in pack["episodes"]:
                start = cum / FPS
                cum += e["length"]
                m[e["episode_index"]] = (fi, start, cum / FPS)
        cam_locs[cam] = m

        i = 0
        while i < len(packs):
            sf = packs[i]["src_file"]
            j = i
            while j < len(packs) and packs[j]["src_file"] == sf:
                j += 1
            group = packs[i:j]
            jobs.append({
                "cam": cam,
                "cam_key": k,
                "src_file": sf,
                "group": group,
                "start_index": i,
                "frames": sum(p["frames"] for p in group),
            })
            i = j
    return jobs, cam_locs


def _run_ffmpeg(cmd: list[str], expected_frames: int,
                progress: Progress | None, note: str,
                progress_cb=None) -> None:
    """Run ffmpeg, feeding `-progress pipe:1` frame counts into the reporter."""
    last = 0
    with tempfile.TemporaryFile(mode="w+") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("frame="):
                continue
            try:
                n = int(line.split("=", 1)[1])
            except ValueError:
                continue
            if n > last:
                delta = n - last
                if progress is not None:
                    progress.update(delta, note=note)
                if progress_cb is not None:
                    progress_cb(delta, note)
                last = n
        rc = proc.wait()
        errf.seek(0)
        err = errf.read()
    if rc != 0:
        tail = "\n".join(err.strip().splitlines()[-20:])
        raise RuntimeError(f"ffmpeg failed (rc={rc}): {tail}")
    # in case progress parsing missed a tail (e.g. segment muxer counters)
    if expected_frames > last:
        delta = expected_frames - last
        if progress is not None:
            progress.update(delta, note=note)
        if progress_cb is not None:
            progress_cb(delta, note)


def transcode_job(task_dir: Path, out_dir: Path, job: dict,
                  src_hw: tuple[int, int], progress: Progress | None,
                  progress_cb=None) -> dict:
    """Transcode ONE source video file into its pre-planned output pack files.

    Safe to run concurrently with other jobs: output file names are disjoint
    and were fixed by build_video_jobs() before any worker started.
    """
    cam = job["cam"]
    cam_key = job["cam_key"]
    sf = job["src_file"]
    group = job["group"]
    start = job["start_index"]
    expected_frames = job["frames"]
    dst_dir = out_dir / "videos" / f"observation.images.{cam}" / "chunk-000"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = task_dir / "videos" / cam_key / "chunk-000" / f"file-{sf:03d}.mp4"
    if not src.exists():
        raise FileNotFoundError(f"missing source video: {src}")
    note = f"{cam} src-file-{sf:03d}"
    t0 = time.time()

    if len(group) > 1:
        bounds, cum = [], 0
        for g in group[:-1]:
            cum += g["frames"]
            bounds.append(cum)
        # force keyframes ONLY at the split boundaries (eq, not gte: gte(n,0)
        # is true for every frame and would force all-intra encoding)
        expr = "expr:eq(n,0)" + "".join(f"+eq(n,{b})" for b in bounds)
        cmd = [FFMPEG, "-v", "error", "-y", "-progress", "pipe:1", "-nostats",
               "-i", str(src),
               "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p",
               "-force_key_frames", expr,
               "-f", "segment",
               "-segment_frames", ",".join(map(str, bounds)),
               "-segment_start_number", str(start),
               "-reset_timestamps", "1",
               str(dst_dir / "file-%03d.mp4")]
    else:
        cmd = [FFMPEG, "-v", "error", "-y", "-progress", "pipe:1", "-nostats",
               "-i", str(src),
               "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p", str(dst_dir / f"file-{start:03d}.mp4")]
    _run_ffmpeg(cmd, expected_frames, progress, note, progress_cb=progress_cb)

    # +faststart post-pass (the segment muxer ignores -movflags; stream-copy
    # each finished file once to move moov to the front, per spec §5)
    for k in range(len(group)):
        p = dst_dir / f"file-{start + k:03d}.mp4"
        tmp = p.with_name(p.stem + ".fs.tmp.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(p),
                        "-c", "copy", "-movflags", "+faststart", str(tmp)],
                       check=True)
        os.replace(tmp, p)

    # verify each produced file
    max_bytes = total_bytes = 0
    for k, pack in enumerate(group):
        p = dst_dir / f"file-{start + k:03d}.mp4"
        info = probe_video(p)
        exp_dur = pack["frames"] / FPS
        if info["nb_frames"] != pack["frames"]:
            raise RuntimeError(f"{p.name}: {info['nb_frames']} frames != expected "
                               f"{pack['frames']}")
        if abs(info["duration"] - exp_dur) > 0.15:
            raise RuntimeError(f"{p.name}: duration {info['duration']:.3f}s != "
                               f"expected {exp_dur:.3f}s")
        if (info["height"], info["width"]) != src_hw:   # src_hw = (h, w)
            raise RuntimeError(f"{p.name}: {info['width']}x{info['height']} != "
                               f"source {src_hw[1]}x{src_hw[0]}")
        if not moov_at_front(p):
            raise RuntimeError(f"{p.name}: moov not at front (+faststart failed)")
        sz = p.stat().st_size
        max_bytes = max(max_bytes, sz)
        total_bytes += sz

    return {"cam": cam, "src_file": sf, "start_index": start,
            "n_files": len(group), "frames": expected_frames,
            "max_bytes": max_bytes, "total_bytes": total_bytes,
            "seconds": time.time() - t0}


def write_episodes_index(ep_meta: list[dict], data_locs: list[dict],
                         cam_locs: dict[str, dict[int, tuple[int, float, float]]],
                         cams: list[str], path: Path) -> None:
    """meta/episodes/chunk-000/file-000.parquet (v3.0 full index).

    cam_locs[cam][episode_index] = (file_index, from_timestamp, to_timestamp)
    with the per-file cumulative timeline (each output file starts at 0).
    """
    loc = {r["episode_index"]: r for r in data_locs}
    n = len(ep_meta)
    cols: dict[str, pa.Array] = {
        "episode_index": pa.array([e["episode_index"] for e in ep_meta], type=pa.int64()),
        "tasks": pa.array([e["tasks"] for e in ep_meta], type=pa.list_(pa.string())),
        "length": pa.array([e["length"] for e in ep_meta], type=pa.int64()),
        "task_index": pa.array([0] * n, type=pa.int64()),
        "task": pa.array([e["tasks"][0] for e in ep_meta], type=pa.string()),
        "meta/episodes/chunk_index": pa.array([0] * n, type=pa.int64()),
        "meta/episodes/file_index": pa.array([0] * n, type=pa.int64()),
        "data/chunk_index": pa.array([loc[e["episode_index"]]["data/chunk_index"] for e in ep_meta],
                                     type=pa.int64()),
        "data/file_index": pa.array([loc[e["episode_index"]]["data/file_index"] for e in ep_meta],
                                    type=pa.int64()),
        "dataset_from_index": pa.array([loc[e["episode_index"]]["dataset_from_index"] for e in ep_meta],
                                       type=pa.int64()),
        "dataset_to_index": pa.array([loc[e["episode_index"]]["dataset_to_index"] for e in ep_meta],
                                     type=pa.int64()),
    }
    for cam in cams:
        key = f"observation.images.{cam}"
        fi, ft, tt = [], [], []
        for e in ep_meta:
            a, b, c = cam_locs[cam][e["episode_index"]]
            fi.append(a)
            ft.append(b)
            tt.append(c)
        cols[f"videos/{key}/chunk_index"] = pa.array([0] * n, type=pa.int64())
        cols[f"videos/{key}/file_index"] = pa.array(fi, type=pa.int64())
        cols[f"videos/{key}/from_timestamp"] = pa.array(ft, type=pa.float64())
        cols[f"videos/{key}/to_timestamp"] = pa.array(tt, type=pa.float64())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    tmp.replace(path)   # atomic swap (spec §4.3: tmp+rename)


def build_features(cams: list[str], video_hw: dict[str, tuple[int, int]],
                   passthrough_types: dict[str, tuple[str, list[int]]]) -> dict:
    f: dict[str, dict] = {}
    for name in CANON_COLS + ["index"]:
        if name == "timestamp":
            f[name] = {"dtype": "float64", "shape": [1], "names": None}
        elif name in INT_COLS:
            f[name] = {"dtype": "int64", "shape": [1], "names": None}
        else:  # the five 55D float32 columns
            f[name] = {"dtype": "float32", "shape": [N_DIM], "names": None}
    for name, (dt, shape) in passthrough_types.items():
        f[name] = {"dtype": dt, "shape": shape, "names": None}
    for cam in cams:
        h, w = video_hw[f"observation.images.{cam}"]
        f[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "rgb"],
            "info": {"video.height": h, "video.width": w, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "video.fps": FPS, "video.channels": 3, "has_audio": False},
        }
    return f


def convert_task(task_dir: Path, out_root: Path, show_progress: bool = True,
                 progress_interval: float = 10.0, video_workers: int = 3,
                 progress_cb=None) -> dict:
    task_name = task_dir.name
    out_dir = out_root / task_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ---------------- read source (lerobot v3.0) ----------------
    src_info = json.loads((task_dir / "meta" / "info.json").read_text())
    tt = pq.read_table(task_dir / "meta" / "tasks.parquet")
    # task string lives either in a 'task' column (our layout) or in the stored
    # pandas index column (e.g. '__index_level_0__', the source layout)
    task_col = "task" if "task" in tt.column_names else next(
        c for c in tt.column_names if c != "task_index")
    tasks = [{"task_index": int(i), "task": str(t)} for i, t
             in zip(tt.column("task_index").to_pylist(), tt.column(task_col).to_pylist())]

    ep_files = sorted((task_dir / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    src_epi = pa.concat_tables([pq.read_table(f) for f in ep_files])
    ep_meta = []
    for row in src_epi.to_pylist():
        ep_meta.append({
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "tasks": list(row["tasks"]),
            "dataset_from_index": int(row["dataset_from_index"]),
            "dataset_to_index": int(row["dataset_to_index"]),
            **{k: int(row[k]) for k in row if k.startswith("videos/") and k.endswith("file_index")},
        })
    ep_meta.sort(key=lambda r: r["episode_index"])

    # source data (one or more files, concatenated in order)
    data_files = sorted((task_dir / "data" / "chunk-000").glob("file-*.parquet"))
    src_tbl = pa.concat_tables([pq.read_table(f) for f in data_files])
    total_frames = sum(e["length"] for e in ep_meta)
    assert src_tbl.num_rows == total_frames, f"data rows {src_tbl.num_rows} != {total_frames}"
    # episode contiguity
    pos = 0
    for e in ep_meta:
        assert e["dataset_from_index"] == pos and \
               e["dataset_to_index"] - e["dataset_from_index"] == e["length"], \
               f"episode {e['episode_index']} index range inconsistent"
        pos = e["dataset_to_index"]
    assert pos == total_frames

    # cameras: source keys like images.rgb.head -> output observation.images.head
    cam_keys = sorted(d.name for d in (task_dir / "videos").iterdir() if d.is_dir())
    cams = [k.split(".")[-1] for k in cam_keys]
    video_hw = {}
    for k, cam in zip(cam_keys, cams):
        feat = src_info["features"][k]
        h, w = feat["shape"][:2]
        video_hw[f"observation.images.{cam}"] = (h, w)

    log(f"  {task_name}: eps={len(ep_meta)} frames={total_frames} "
        f"cams={cams} task={tasks[0]['task']!r}")

    # passthrough column types from source schema (float32 scalar or list)
    passthrough_types: dict[str, tuple[str, list[int]]] = {}
    for name in PASSTHROUGH_COLS:
        t = src_tbl.schema.field(name).type
        if pa.types.is_fixed_size_list(t):
            passthrough_types[name] = ("float32", [t.list_size])
        else:
            passthrough_types[name] = ("float32", [1])   # scalar (shape-[1] feature)

    # ---------------- build episode tables (55D + passthrough) ----------------
    data_prog = Progress(total_frames, "frames", f"data:{task_name}",
                         enabled=show_progress, interval=progress_interval)
    data_prog.start_heartbeat()
    ep_tables: list[tuple[dict, pa.Table]] = []
    cursor = 0
    for e in ep_meta:
        sub = src_tbl.slice(e["dataset_from_index"], e["length"])
        tbl = build_episode_table(sub, e["episode_index"], cursor)
        cursor += tbl.num_rows
        ep_tables.append((e, tbl))
        data_prog.update(e["length"], note=f"ep {e['episode_index']}")
        if progress_cb is not None:
            progress_cb(e["length"], f"data ep {e['episode_index']}")
    data_prog.finish()

    # in-convert sanity: FK(joints) vs the 55D eef we just wrote (3 frames)
    st = np.stack([ep_tables[0][1].column("state_unified").to_pylist()[0],
                   ep_tables[len(ep_tables) // 2][1].column("state_unified").to_pylist()[0],
                   ep_tables[-1][1].column("state_unified").to_pylist()[-1]]).astype(np.float64)
    jj = st[:, LEFT_ARM_JOINT[0]:LEFT_ARM_JOINT[1]]
    pos_fk, quat_fk = fk_flange(jj)
    pos_diff = np.abs(pos_fk - st[:, LEFT_EEF[0]:LEFT_EEF[0] + 3]).max()
    q_diff = np.abs(np.minimum(np.abs(quat_fk - st[:, LEFT_EEF[0] + 3:LEFT_EEF[1]]),
                               np.abs(quat_fk + st[:, LEFT_EEF[0] + 3:LEFT_EEF[1]]))).max()
    log(f"    FK sanity (3 frames): pos diff {pos_diff:.2e} m, quat diff {q_diff:.2e}")
    assert pos_diff < 1e-4 and q_diff < 1e-3, "FK cross-check failed in convert"

    # ---------------- data parquet (~100MB rotation) ----------------
    data_locs = write_data(ep_tables, out_dir / "data" / "chunk-000")

    # ---------------- videos: transcode + <=960s packing (parallel) ----------
    # One job per source video file; jobs write disjoint pre-planned output
    # files, so they can run concurrently without changing file numbering.
    jobs, cam_locs = build_video_jobs(ep_meta, cam_keys, cams)
    for cam in cams:
        assert len(cam_locs[cam]) == len(ep_meta), f"{cam}: packing lost episodes"
    video_total = sum(j["frames"] for j in jobs)
    assert video_total == total_frames * len(cams), \
        f"video frames {video_total} != {total_frames} x {len(cams)}"
    video_prog = Progress(video_total, "frames", f"video:{task_name}",
                          enabled=show_progress, interval=progress_interval)
    video_prog.start_heartbeat()

    video_stats: dict[str, dict] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(video_workers))) as ex:
        futs = {ex.submit(transcode_job, task_dir, out_dir, j,
                          video_hw[f"observation.images.{j['cam']}"],
                          video_prog, progress_cb): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                info = fut.result()
            except Exception as e:  # noqa: BLE001
                msg = f"{j['cam']} src-file-{j['src_file']:03d}: {type(e).__name__}: {e}"
                errors.append(msg)
                log(f"    [FAIL] {msg}")
                continue
            s = video_stats.setdefault(j["cam"], {"n_files": 0, "max_bytes": 0,
                                                  "total_bytes": 0})
            s["n_files"] += info["n_files"]
            s["max_bytes"] = max(s["max_bytes"], info["max_bytes"])
            s["total_bytes"] += info["total_bytes"]
            log(f"    {j['cam']}: source file-{j['src_file']:03d}.mp4 -> "
                f"file-{j['start_index']:03d}.."
                f"{j['start_index'] + len(j['group']) - 1:03d}.mp4 "
                f"({info['n_files']} files, {info['frames'] / FPS:.0f}s, "
                f"{info['seconds']:.0f}s)")
    video_prog.finish(ok=not errors)
    if errors:
        raise RuntimeError("video transcode failed: " + "; ".join(errors))

    # ---------------- meta ----------------
    meta = out_dir / "meta"
    write_vocab_parquet(meta / "tasks.parquet", "task",
                        [(t["task_index"], t["task"]) for t in tasks])
    write_vocab_parquet(meta / "subtasks.parquet", "subtask", [])  # empty: no labels
    write_episodes_index(ep_meta, data_locs, cam_locs, cams,
                         meta / "episodes" / "chunk-000" / "file-000.parquet")

    with open(meta / "episodes.jsonl", "w", encoding="utf-8") as f:
        for e in ep_meta:
            f.write(json.dumps({
                "episode_index": e["episode_index"],
                "tasks": e["tasks"],
                "length": e["length"],
                "action_config": [],                       # no subtask annotations
                "source_fps": float(FPS),
                "source_duration_s": e["length"] / FPS,
            }, ensure_ascii=False) + "\n")

    tot_mb = sum(s["total_bytes"] for s in video_stats.values()) / 1e6
    max_mb = max(s["max_bytes"] for s in video_stats.values()) / 1e6
    n_videos = sum(s["n_files"] for s in video_stats.values())

    info = {
        "codebase_version": "v3.0",
        "robot_type": src_info.get("robot_type", "Franka"),
        "total_episodes": len(ep_meta),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(ep_meta)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": build_features(cams, video_hw, passthrough_types),
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "custom": {
            "converter": "intern_convert.py (lerobot v3.0 -> v3.0 + 55D)",
            "source": str(task_dir),
            "mapping_55d": {
                "left_arm_joint": "single 7-DOF Franka arm -> left slot 0:7 from "
                                  "states.joint.position (radians); right arm 7:14 empty",
                "left_eef": "flange pose from states.gripper.pose: position (m) + euler "
                            "(URDF rpy) -> quaternion xyzw; robot base frame ('armbase'); "
                            "verified against standard Franka DH forward kinematics "
                            "(0.0000 deg / 0.00000 m over all 140157 frames); right eef "
                            "21:28 empty",
                "left_gripper": "raw gripper opening from states.gripper.position "
                                "(meters, [0, 0.08]; NOT normalized); right gripper empty",
                "action": "next-frame state (last frame copies itself), in-episode only; "
                          "source actions.* (command targets) and master_actions.* (teleop "
                          "master arm) kept as passthrough columns",
                "subtasks": "no subtask labels in source; subtask_index == -1, "
                            "subtasks.parquet empty",
            },
            "coordinate_frames": {
                "eef": "flange (tool mounting face) pose in the robot base frame "
                       "('armbase'; base offset vs standard Franka DH base verified zero)",
                "passthrough_quats": "states.ee_to_* / states.tcp_to_* / actions.* poses "
                                     "keep the source's ORIGINAL quaternion values verbatim. "
                                     "NOTE: the source's ee/tcp quaternion encoding is "
                                     "defective (no standard convention x-/w-first, "
                                     "normal/inverse, axis-permutation reproduces the true "
                                     "orientation); these columns are informational only and "
                                     "were NOT used for the 55D eef slot",
                "gripper_pose": "states.gripper.pose = flange position + euler (URDF rpy, "
                                "scipy from_euler('xyz')); this is the source of the 55D eef",
            },
            "passthrough": {n: "kept verbatim from source" for n in PASSTHROUGH_COLS},
            "time_axis": {
                "container_fps": FPS,
                "note": "source is natively 30fps; timestamp = frame_index/30. No "
                        "resampling, no source_timestamp column needed.",
            },
            "videos": {
                "mode": "av1 -> h264 transcode + <=960s episode-boundary packing "
                        "(one ffmpeg pass per source file; spec §5)",
                "encode": "libx264 crf=18 preset=fast yuv420p +faststart",
                "n_videos": n_videos,
                "per_camera": {cam: {"n_files": s["n_files"],
                                     "max_file_mb": round(s["max_bytes"] / 1e6, 2)}
                               for cam, s in video_stats.items()},
                "total_mb": round(tot_mb, 1),
                "max_file_mb": round(max_mb, 2),
                "from_to_timestamp": "per-file cumulative timeline (each file starts at 0); "
                                     "all files <=960s (28800 frames)",
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    # modality.json (informational; mirrors mapping_55d)
    eef_note = {"original_key": "states.gripper.pose (pos + euler rpy -> quat xyzw)",
                "absolute": True,
                "note": "flange (tool mounting face), robot base frame, pos(3)+quat xyzw(4)"}
    modality = {
        "state": {
            "left_arm_joint": {"original_key": SRC_JOINT, "start": 0, "end": 7, "absolute": True},
            "left_eef": dict(eef_note),
            "left_gripper": {"original_key": SRC_GRIPPER, "start": 0, "end": 1, "absolute": True},
        },
        "action": {
            "left_arm_joint": {"original_key": "derived: next-frame state", "absolute": True},
            "left_eef": {"original_key": "derived: next-frame state", "absolute": True},
            "left_gripper": {"original_key": "derived: next-frame state", "absolute": True},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2, ensure_ascii=False))

    # qc_report.json (L1 QC)
    data_file_names = sorted(p.name for p in (out_dir / "data" / "chunk-000").glob("file-*.parquet"))
    qc = {
        "task": task_name,
        "total_episodes": len(ep_meta),
        "total_frames": total_frames,
        "total_videos": n_videos,
        "data_files": data_file_names,
        "mask_active_slots": ACTIVE_SLOTS,
        "mask_note": "left arm joints 0:7 + left eef 14:21 (flange from gripper.pose) + "
                     "left gripper 28; all other slots mask=0 (single-arm robot)",
        "action_convention": "action_unified[t] = state_unified[t+1] (last frame copies itself)",
        "video_note": "transcoded av1->h264, packed <=960s at episode boundaries; "
                      "source ee/tcp quaternions kept verbatim in passthrough (defective; "
                      "not used for 55D eef)",
    }
    (out_dir / "qc_report.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False))

    # process marker (last write)
    (out_dir / ".convert_done").write_text(json.dumps({
        "task": task_name, "episodes": len(ep_meta), "frames": total_frames,
        "videos": n_videos, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False))

    return {"task": task_name, "episodes": len(ep_meta), "frames": total_frames,
            "videos": n_videos, "out": str(out_dir)}


def discover_datasets(src_root: Path) -> list[Path]:
    """All dataset dirs (containing meta/info.json) under src_root, any depth."""
    found: list[Path] = []
    for root, dirs, _files in os.walk(src_root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        rp = Path(root)
        if (rp / "meta" / "info.json").is_file():
            found.append(rp)
            dirs[:] = []
    return found


def _convert_worker(dataset_dir: str, out_root: str, video_workers: int,
                    progress_q=None) -> dict:
    """Worker-process entry point: convert one dataset, capture its log."""
    import pyarrow as pa  # local import: safe in spawned/forked workers
    pa.set_cpu_count(1)     # each worker writes its own parquet; avoid oversubscription

    def cb(n: float, note: str) -> None:
        if progress_q is not None:
            progress_q.put((float(n), str(note)))

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            r = convert_task(Path(dataset_dir), Path(out_root),
                             show_progress=False, video_workers=video_workers,
                             progress_cb=cb)
        return {"ok": True, "result": r, "log": buf.getvalue()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "log": buf.getvalue()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="/data5T/liujin/dataset/InternData-A1",
                    help="root scanned recursively for dataset dirs (meta/info.json); "
                         "run intern_extract.py first to materialize extracted archives")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent / "processed_dataset"))
    ap.add_argument("--task", nargs="+", default=None,
                    help="dataset dir name(s), e.g. close_the_laptop (default: all discovered)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip datasets that already have a .convert_done marker")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep converting remaining datasets after one fails")
    ap.add_argument("--workers", type=int, default=8,
                    help="datasets converted in parallel (1 = sequential; default: 8; "
                         "note each worker also runs --video-workers ffmpeg jobs, "
                         "so total ffmpeg concurrency = workers x video-workers)")
    ap.add_argument("--video-workers", type=int, default=3,
                    help="concurrent ffmpeg transcode jobs per dataset (default: 3)")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable progress/ETA lines")
    ap.add_argument("--progress-interval", type=float, default=10.0,
                    help="seconds between progress lines (default: 10)")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(src_root)
    if args.task:
        wanted = set(args.task)
        datasets = [d for d in datasets if d.name in wanted]
        missing = wanted - {d.name for d in datasets}
        if missing:
            log(f"WARN: requested dataset(s) not found under {src_root}: {sorted(missing)}")
    if not datasets:
        log(f"no dataset dirs (with meta/info.json) found under {src_root} "
            f"(extract archives first with intern_extract.py)")
        sys.exit(1)
    names = [d.name for d in datasets]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        sys.exit(f"duplicate dataset dir name(s) {dupes} under {src_root} would "
                 f"collide in {out_root}; rename or extract them separately")

    # --skip-existing: completed datasets are removed from both work and ETA totals
    pending: list[Path] = []
    n_skipped = 0
    for d in datasets:
        if args.skip_existing and (out_root / d.name / ".convert_done").exists():
            log(f"[SKIP] {d.name}: already converted (.convert_done present)")
            n_skipped += 1
        else:
            pending.append(d)

    frames_by_ds: dict[str, int] = {}
    work_by_ds: dict[str, int] = {}
    for d in pending:
        try:
            frames_by_ds[d.name] = dataset_frame_count(d)
            work_by_ds[d.name] = dataset_work_units(d)
        except Exception as e:  # noqa: BLE001  (broken dataset must not abort the batch)
            log(f"WARN: cannot pre-scan {d.name}: {e}")
            frames_by_ds[d.name] = 0
            work_by_ds[d.name] = 0
    total_frames = sum(frames_by_ds.values())
    total_work = sum(work_by_ds.values())
    prog = Progress(total_work, "frames", "convert",
                    enabled=not args.no_progress,
                    interval=args.progress_interval)
    prog.start_heartbeat()

    log(f"TASK_START total={len(datasets)} pending={len(pending)} "
        f"skipped={n_skipped} frames={total_frames:,}")
    results: list[dict] = []
    failed: list[str] = []
    workers = max(1, int(args.workers))
    video_workers = max(1, int(args.video_workers))

    if workers == 1:
        for d in pending:
            t = d.name
            r = None
            try:
                r = convert_task(d, out_root,
                                 show_progress=not args.no_progress,
                                 progress_interval=args.progress_interval,
                                 video_workers=video_workers,
                                 progress_cb=lambda n, note: prog.update(n, note=note))
                results.append(r)
                log(f"[OK] {t}: {r['episodes']} eps, {r['frames']} frames, "
                    f"{r['videos']} videos")
            except Exception as e:  # noqa: BLE001
                log(f"[FAIL] {t}: {e}")
                failed.append(t)
                if not args.continue_on_error:
                    break
    else:
        mgr = mp.Manager()
        q = mgr.Queue()
        stop = threading.Event()
        fwd = start_queue_forwarder(q, prog, stop)
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_convert_worker, str(d), str(out_root),
                                  video_workers, q): d for d in pending}
                cancelled = False
                for fut in as_completed(futs):
                    d = futs[fut]
                    t = d.name
                    if cancelled:
                        continue
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa: BLE001  (worker crash/pickling)
                        res = {"ok": False, "error": f"{type(e).__name__}: {e}",
                               "log": ""}
                    if res.get("log"):
                        for line in res["log"].splitlines():
                            log(f"[{t}] {line}")
                    if res["ok"]:
                        r = res["result"]
                        results.append(r)
                        log(f"[OK] {t}: {r['episodes']} eps, {r['frames']} frames, "
                            f"{r['videos']} videos")
                    else:
                        log(f"[FAIL] {t}: {res['error']}")
                        failed.append(t)
                        if not args.continue_on_error:
                            cancelled = True
                            for f2 in futs:
                                f2.cancel()
        finally:
            q.put(None)
            stop.set()
            fwd.join(timeout=2)
            mgr.shutdown()

    prog.finish(ok=not failed)
    log(f"TASK_DONE ok={len(results)} skipped={n_skipped} failed={len(failed)}")
    if failed:
        log("FAILED_TASKS: " + " ".join(failed))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
