#!/usr/bin/env python
"""GM100 (songling_selfcollect) lerobot v2.1 -> unified 55D LeRobot v3.0.

Converts the dual-arm (6-DOF x 2) "gm100-cobotmagic" demo dataset from the old
per-episode v2.1 layout into the team's unified v3.0 format
(see standard_format/README.md):

  src (v2.1)                                dst (v3.0 + 55D)
    task_XXXXXX/                              task_XXXXXX/
      data/chunk-000/episode_*.parquet         data/chunk-000/file-*.parquet    (~100MB rotate)
      videos/chunk-000/<key>/episode_*.mp4     videos/<key>/chunk-000/file-*.mp4 (hard link)
      meta/info.json  (v2.1)                   meta/info.json  (v3.0)
      meta/tasks.jsonl                         meta/tasks.parquet
      meta/episodes.jsonl                      meta/subtasks.parquet (empty: no labels)
                                               meta/episodes.jsonl
                                               meta/episodes/chunk-000/file-*.parquet
                                               meta/modality.json
                                               qc_report.json

55D slot mapping (55 = L/R arm-joint 7+7, eef 7+7, gripper 1+1, hand 6+6,
waist 4, head 2, mobile 3, reserved 4):
  0:6    <- left  arm 6 joints           (mask 1; slot 6 empty, 6-DOF arm in 7-wide slot)
  7:13   <- right arm 6 joints           (mask 1; slot 13 empty)
  14:28  <- eef via forward kinematics   (mask 1): wrist-flange (link6) pose of the
            Piper arm (urdf/aloha_tracer2_dabai_dark.urdf; see gm100_fk.py),
            position 3 + quat xyzw 4, in the robot base frame
            (URDF root 'footprint', coincident with the arm-mount body_Link)
  28:30  <- left/right gripper           (mask 1; raw opening, NOT normalized)
  30:55  <- EMPTY (mask 0)               (hand/waist/head/mobile/reserved)

action_unified[t] = state_unified[t+1] (last frame copies itself), strictly
within each episode. Source `action.arm.position` (command target) is kept as
a passthrough column, NOT used as action_unified (audited convention).

Passthrough columns (kept verbatim; "可以不处理，但不能丢"):
  observation.state.arm.velocity, observation.state.arm.effort,
  observation.state.effector.effort, action.arm.position, action.effector.position

Data rotation: matches lerobot's own metric (`data_files_size_in_mb` compares the
on-disk size of the current parquet, lerobot_dataset.py). We estimate on-disk
size from in-memory bytes with a self-calibrating compression ratio (prior 5.0,
updated from each flush), and rotate at episode boundaries only.

Usage (cosmos-framework env):
  conda activate cosmos-framework
  python gm100_convert.py --task task_00001            # pilot
  python gm100_convert.py                              # all task_* dirs
"""
from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gm100_fk import GM100FK, DEFAULT_URDF  # noqa: E402
from _progress import Progress, start_queue_forwarder  # noqa: E402

FPS = 30
DATA_ROTATE_BYTES = 100 * 1024 * 1024   # on-disk target per data file
COMPRESSION_PRIOR = 5.0                 # in-mem/disk prior before first flush

# 55D slot layout (standard_format/README.md §3.2)
N_DIM = 55
LEFT_ARM_JOINT = (0, 6)      # slots 0..5; slot 6 stays empty (7-wide slot, 6-DOF arm)
RIGHT_ARM_JOINT = (7, 13)    # slots 7..12; slot 13 stays empty
LEFT_EEF = (14, 21)          # 3 pos + 4 quat xyzw, filled via FK (gm100_fk.py)
RIGHT_EEF = (21, 28)
LEFT_GRIPPER = 28
RIGHT_GRIPPER = 29
ACTIVE_SLOTS = sorted(list(range(*LEFT_ARM_JOINT)) +
                      list(range(*RIGHT_ARM_JOINT)) +
                      list(range(*LEFT_EEF)) +
                      list(range(*RIGHT_EEF)) +
                      [LEFT_GRIPPER, RIGHT_GRIPPER])

# source column names (v2.1) and their roles
SRC_ARM_POS = "observation.state.arm.position"       # (12) -> joint slots
SRC_EFF_POS = "observation.state.effector.position"  # (2)  -> gripper slots
PASSTHROUGH_COLS = [
    "observation.state.arm.velocity",                # (12)
    "observation.state.arm.effort",                  # (12)
    "observation.state.effector.effort",             # (2)
    "action.arm.position",                           # (12) command target
    "action.effector.position",                      # (2)  command target
]

# final data-column order (11 canonical + 5 passthrough + global index)
CANON_COLS = [
    "frame_index", "episode_index", "timestamp", "task_index",
    "state_unified", "action_unified", "mask_state", "mask_action", "reliability",
    "subtask_index",
]
DATA_COL_ORDER = CANON_COLS + PASSTHROUGH_COLS + ["index"]

INT_COLS = {"frame_index", "episode_index", "task_index", "subtask_index", "index"}


def log(line: str) -> None:
    print(line, flush=True)


def fixed_list_f32(arr: np.ndarray) -> pa.Array:
    """(N, D) float32 ndarray -> FixedSizeListArray<float32>[D]."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(arr.ravel(), type=pa.float32()), arr.shape[1])


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_frame_count(src_root: Path, task: str) -> int:
    """Total frames of one source task (cheap: episodes.jsonl only)."""
    return sum(int(r["length"]) for r in
               read_jsonl(src_root / task / "meta" / "episodes.jsonl"))


def build_episode_table(src: pa.Table, ep_idx: int, global_cursor: int,
                        fk) -> pa.Table:
    """Build the 16-column v3.0 table for one episode (source is one parquet).

    `fk` is a gm100_fk.GM100FK instance; eef slots are filled via FK.
    """
    T = src.num_rows
    arm_pos = np.stack(src.column(SRC_ARM_POS).to_pylist()).astype(np.float32)   # (T,12)
    eff_pos = np.stack(src.column(SRC_EFF_POS).to_pylist()).astype(np.float32)   # (T,2)
    assert arm_pos.shape[1] == 12, f"arm.position width {arm_pos.shape[1]} != 12"
    assert eff_pos.shape[1] == 2, f"effector.position width {eff_pos.shape[1]} != 2"

    # state_unified
    state = np.zeros((T, N_DIM), dtype=np.float32)
    state[:, LEFT_ARM_JOINT[0]:LEFT_ARM_JOINT[1]] = arm_pos[:, 0:6]
    state[:, RIGHT_ARM_JOINT[0]:RIGHT_ARM_JOINT[1]] = arm_pos[:, 6:12]
    state[:, LEFT_GRIPPER] = eff_pos[:, 0]
    state[:, RIGHT_GRIPPER] = eff_pos[:, 1]

    # eef via forward kinematics (Piper arm URDF), robot base frame
    pos_l, quat_l = fk.fk_left(arm_pos[:, 0:6])     # (T,3) m, (T,4) xyzw
    pos_r, quat_r = fk.fk_right(arm_pos[:, 6:12])
    state[:, LEFT_EEF[0]:LEFT_EEF[0] + 3] = pos_l
    state[:, LEFT_EEF[0] + 3:LEFT_EEF[1]] = quat_l
    state[:, RIGHT_EEF[0]:RIGHT_EEF[0] + 3] = pos_r
    state[:, RIGHT_EEF[0] + 3:RIGHT_EEF[1]] = quat_r

    # action_unified = next-frame state (last frame copies itself), in-episode only
    action = np.zeros((T, N_DIM), dtype=np.float32)
    action[0:T - 1] = state[1:T]
    action[T - 1] = state[T - 1]

    # mask / reliability: which 55D slots carry real data (reliability == mask)
    mask = np.zeros((T, N_DIM), dtype=np.float32)
    mask[:, LEFT_ARM_JOINT[0]:LEFT_ARM_JOINT[1]] = 1
    mask[:, RIGHT_ARM_JOINT[0]:RIGHT_ARM_JOINT[1]] = 1
    mask[:, LEFT_EEF[0]:RIGHT_EEF[1]] = 1
    mask[:, LEFT_GRIPPER:RIGHT_GRIPPER + 1] = 1

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
        arr = np.stack(src.column(name).to_pylist()).astype(np.float32)
        cols[name] = fixed_list_f32(arr)
    cols["index"] = pa.array(np.arange(global_cursor, global_cursor + T, dtype=np.int64))

    return pa.table({k: cols[k] for k in DATA_COL_ORDER})


def write_data(ep_tables: list[tuple[dict, pa.Table]], data_dir: Path) -> list[dict]:
    """Concatenate episode tables, rotating into file-XXX.parquet at episode
    boundaries. Rotation follows lerobot's metric: the on-disk size of the
    current file (data_files_size_in_mb). On-disk size is estimated from
    in-memory bytes via a self-calibrating compression ratio.

    Returns per-episode data-location rows.
    """
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
            disk_per_mem = disk / pending_bytes  # calibrate for later files
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
    """Write tasks.parquet / subtasks.parquet in the official layout
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


def write_episodes_index(ep_meta: list[dict], data_locs: list[dict], cams: list[str],
                         path: Path) -> None:
    """Write meta/episodes/chunk-000/file-000.parquet (v3.0 full index)."""
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
        cols[f"videos/{key}/chunk_index"] = pa.array([0] * n, type=pa.int64())
        cols[f"videos/{key}/file_index"] = pa.array([e["episode_index"] for e in ep_meta],
                                                    type=pa.int64())
        cols[f"videos/{key}/from_timestamp"] = pa.array([0.0] * n, type=pa.float64())
        cols[f"videos/{key}/to_timestamp"] = pa.array([e["length"] / FPS for e in ep_meta],
                                                      type=pa.float64())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path, compression="zstd")


def link_videos(task_dir: Path, out_dir: Path, ep_meta: list[dict], cams: list[str]) -> dict:
    """Link per-episode mp4 (v2.1 layout -> v3.0 layout).

    src: videos/chunk-000/<key>/episode_XXXXXX.mp4
    dst: videos/<key>/chunk-000/file-XXX.mp4   (file_index == episode_index)

    Uses os.link (zero-copy) when src and dst share a filesystem; falls back to
    a real copy when linking fails (e.g. cross-device, EXDEV in multi-disk
    setups) -- in that case the output occupies real disk space.
    """
    max_bytes = 0
    total_bytes = 0
    n_linked = 0
    n_copied = 0
    link_warned = False
    for cam in cams:
        key = f"observation.images.{cam}"
        dst_dir = out_dir / "videos" / key / "chunk-000"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for e in ep_meta:
            ep = e["episode_index"]
            src = task_dir / "videos" / "chunk-000" / key / f"episode_{ep:06d}.mp4"
            if not src.exists():
                raise FileNotFoundError(f"missing source video: {src}")
            dst = dst_dir / f"file-{ep:03d}.mp4"
            try:
                os.link(src, dst)
                n_linked += 1
            except OSError as err:
                # cross-device link or other link failure -> real copy
                if not link_warned:
                    log(f"    WARN: os.link failed ({err}); falling back to file copy "
                        f"(output will occupy real disk space)")
                    link_warned = True
                shutil.copy2(src, dst)
                n_copied += 1
            sz = src.stat().st_size
            max_bytes = max(max_bytes, sz)
            total_bytes += sz
    return {"n_videos": n_linked + n_copied, "n_linked": n_linked,
            "n_copied": n_copied, "max_bytes": max_bytes, "total_bytes": total_bytes}


def build_features(cams: list[str], video_hw: tuple[int, int],
                   passthrough_widths: dict[str, int]) -> dict:
    f: dict[str, dict] = {}
    for name in CANON_COLS + ["index"]:
        if name == "timestamp":
            f[name] = {"dtype": "float64", "shape": [1], "names": None}
        elif name in INT_COLS:
            f[name] = {"dtype": "int64", "shape": [1], "names": None}
        else:  # the five 55D float32 columns
            f[name] = {"dtype": "float32", "shape": [N_DIM], "names": None}
    for name, w in passthrough_widths.items():
        f[name] = {"dtype": "float32", "shape": [w], "names": None}
    h, w = video_hw
    for cam in cams:
        f[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "rgb"],
            "info": {"video.height": h, "video.width": w, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "video.fps": FPS, "video.channels": 3, "has_audio": False},
        }
    return f


def convert_task(src_root: Path, task_name: str, out_root: Path,
                 urdf_path: Path = DEFAULT_URDF, progress_cb=None) -> dict:
    task_dir = src_root / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task dir not found: {task_dir}")
    out_dir = out_root / task_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    src_info = json.loads((task_dir / "meta" / "info.json").read_text())
    tasks = read_jsonl(task_dir / "meta" / "tasks.jsonl")
    ep_meta = sorted(read_jsonl(task_dir / "meta" / "episodes.jsonl"),
                     key=lambda r: r["episode_index"])
    cams = sorted(p.name.replace("observation.images.", "")
                  for p in (task_dir / "videos" / "chunk-000").iterdir() if p.is_dir())
    # video resolution from source info.json
    h, w = src_info["features"][f"observation.images.{cams[0]}"]["shape"][:2]

    log(f"  {task_name}: eps={len(ep_meta)} cams={cams} video={w}x{h} "
        f"task={tasks[0]['task']!r}")

    fk = GM100FK(urdf_path)  # forward kinematics for eef slots (Piper arm URDF)

    # --- build episode tables (v3.0 columns) with a running global index ---
    ep_tables: list[tuple[dict, pa.Table]] = []
    passthrough_widths: dict[str, int] = {}
    cursor = 0
    for e in ep_meta:
        ep_idx = e["episode_index"]
        src_path = task_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not src_path.exists():
            raise FileNotFoundError(f"missing source parquet: {src_path}")
        src = pq.read_table(src_path)
        if src.num_rows != e["length"]:
            raise RuntimeError(f"episode {ep_idx}: parquet rows {src.num_rows} "
                               f"!= episodes.jsonl length {e['length']}")
        for name in PASSTHROUGH_COLS:
            wid = len(src.column(name)[0].as_py())
            passthrough_widths.setdefault(name, wid)
            assert passthrough_widths[name] == wid, f"{name} width varies: {wid}"
        tbl = build_episode_table(src, ep_idx, cursor, fk)
        cursor += tbl.num_rows
        ep_tables.append((e, tbl))
        if progress_cb is not None:
            progress_cb(tbl.num_rows, f"{task_name} ep {ep_idx}")
    total_frames = cursor

    # --- data parquet (~100MB on-disk rotation, episode boundaries only) ---
    data_locs = write_data(ep_tables, out_dir / "data" / "chunk-000")

    # --- videos: per-episode hard links (no merge; each file << 960s cap) ---
    vinfo = link_videos(task_dir, out_dir, ep_meta, cams)
    video_files_size_mb = max(1, math.ceil(vinfo["max_bytes"] / 1e6))

    # --- meta ---
    meta = out_dir / "meta"
    write_vocab_parquet(meta / "tasks.parquet", "task",
                        [(t["task_index"], t["task"]) for t in tasks])
    write_vocab_parquet(meta / "subtasks.parquet", "subtask", [])  # empty: no labels
    write_episodes_index(ep_meta, data_locs, cams,
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

    # --- info.json (v3.0) ---
    info = {
        "codebase_version": "v3.0",
        "robot_type": src_info.get("robot_type", "songling_selfcollect"),
        "total_episodes": len(ep_meta),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(ep_meta)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": build_features(cams, (h, w), passthrough_widths),
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": video_files_size_mb,
        "custom": {
            "converter": "gm100_convert.py (lerobot v2.1 -> v3.0 + 55D)",
            "source": str(src_info.get("source", task_dir)),
            "urdf": {
                "file": "urdf/aloha_tracer2_dabai_dark.urdf",
                "source": "https://github.com/agilexrobotics/mobile_aloha_sim @ f799ae0",
                "eef_frame": "wrist flange (link6) rel. to URDF root 'footprint'; "
                             "left arm = fl chain, right arm = fr chain",
            },
            "mapping_55d": {
                "left/right_arm_joint": "6-DOF dual arm; slots 0:6 (left) / 7:13 (right) from "
                                        "observation.state.arm.position (radians); slots 6/13 "
                                        "empty (7-wide slot, 6-DOF arm)",
                "left/right_eef": "wrist-flange (link6) pose from forward kinematics on the arm "
                                  "joints, via urdf/aloha_tracer2_dabai_dark.urdf (Piper arm); "
                                  "position (m, 3) + quaternion xyzw (4); robot base frame "
                                  "(URDF root 'footprint' == arm-mount body_Link); joint order / "
                                  "zero / sign aligned to URDF joint limits (validated)",
                "left/right_gripper": "raw gripper opening from observation.state.effector.position "
                                      "(NOT normalized; downstream normalizes across datasets)",
                "action": "next-frame state (last frame copies itself), in-episode only; source "
                          "action.arm.position (command target) kept as passthrough column",
                "subtasks": "no subtask labels in source; subtask_index == -1, subtasks.parquet empty",
            },
            "passthrough": {n: "kept verbatim from source" for n in PASSTHROUGH_COLS},
            "time_axis": {
                "container_fps": FPS,
                "note": "source is natively 30fps; timestamp = frame_index/30. No resampling, "
                        "no source_timestamp column needed.",
            },
            "videos": {
                "mode": "per-episode files (os.link hard link; auto-fallback to copy "
                        "if src/out cross filesystems)",
                "n_videos": vinfo["n_videos"],
                "n_linked": vinfo["n_linked"],
                "n_copied": vinfo["n_copied"],
                "max_file_mb": round(vinfo["max_bytes"] / 1e6, 2),
                "note": "v2.1 layout videos/chunk-000/<key>/episode_*.mp4 re-linked to v3.0 "
                        "videos/<key>/chunk-000/file-*.mp4 without re-encoding",
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    # --- modality.json (informational; mirrors mapping_55d) ---
    modality = {
        "state": {
            "left_arm_joint": {"original_key": SRC_ARM_POS, "start": 0, "end": 6, "absolute": True},
            "right_arm_joint": {"original_key": SRC_ARM_POS, "start": 6, "end": 12, "absolute": True},
            "left_eef": {"original_key": "derived: FK(arm.position[0:6]) via "
                                        "urdf/aloha_tracer2_dabai_dark.urdf",
                         "absolute": True,
                         "note": "link6 wrist flange, robot base frame, pos(3)+quat xyzw(4)"},
            "right_eef": {"original_key": "derived: FK(arm.position[6:12]) via "
                                         "urdf/aloha_tracer2_dabai_dark.urdf",
                          "absolute": True,
                          "note": "link6 wrist flange, robot base frame, pos(3)+quat xyzw(4)"},
            "left_gripper": {"original_key": SRC_EFF_POS, "start": 0, "end": 1, "absolute": True},
            "right_gripper": {"original_key": SRC_EFF_POS, "start": 1, "end": 2, "absolute": True},
        },
        "action": {
            "left_arm_joint": {"original_key": SRC_ARM_POS, "start": 0, "end": 6, "absolute": True},
            "right_arm_joint": {"original_key": SRC_ARM_POS, "start": 6, "end": 12, "absolute": True},
            "left_eef": {"original_key": "derived: FK(arm.position[0:6]) via "
                                        "urdf/aloha_tracer2_dabai_dark.urdf",
                         "absolute": True,
                         "note": "link6 wrist flange, robot base frame, pos(3)+quat xyzw(4)"},
            "right_eef": {"original_key": "derived: FK(arm.position[6:12]) via "
                                         "urdf/aloha_tracer2_dabai_dark.urdf",
                          "absolute": True,
                          "note": "link6 wrist flange, robot base frame, pos(3)+quat xyzw(4)"},
            "left_gripper": {"original_key": SRC_EFF_POS, "start": 0, "end": 1, "absolute": True},
            "right_gripper": {"original_key": SRC_EFF_POS, "start": 1, "end": 2, "absolute": True},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2, ensure_ascii=False))

    # --- qc_report.json (L1 QC) ---
    data_files = sorted(p.name for p in (out_dir / "data" / "chunk-000").glob("file-*.parquet"))
    qc = {
        "task": task_name,
        "total_episodes": len(ep_meta),
        "total_frames": total_frames,
        "total_videos": vinfo["n_videos"],
        "data_files": data_files,
        "mask_active_slots": ACTIVE_SLOTS,
        "mask_note": "arm joints 0:6 & 7:13 + eef 14:28 (FK) + grippers 28:30; "
                     "hand/waist/head/mobile/reserved all mask=0",
        "action_convention": "action_unified[t] = state_unified[t+1] (last frame copies itself)",
    }
    (out_dir / "qc_report.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False))

    # process marker (last write): makes --skip-existing a safe resume for large
    # batches. Process markers are not part of the deliverable (see standard
    # format doc: ".p2_done" etc. do not get uploaded).
    (out_dir / ".convert_done").write_text(json.dumps({
        "task": task_name, "episodes": len(ep_meta), "frames": total_frames,
        "videos": vinfo["n_videos"], "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False))

    return {"task": task_name, "episodes": len(ep_meta), "frames": total_frames,
            "videos": vinfo["n_videos"], "out": str(out_dir)}


def _convert_worker(src_root: str, task: str, out_root: str, urdf: str,
                    progress_q=None) -> dict:
    """Worker-process entry point: convert one task, capture its log."""
    import pyarrow as pa  # local import: safe in spawned/forked workers
    pa.set_cpu_count(1)     # avoid N workers x pyarrow threads oversubscription

    def cb(n: float, note: str) -> None:
        if progress_q is not None:
            progress_q.put((float(n), str(note)))

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            r = convert_task(Path(src_root), task, Path(out_root), Path(urdf),
                             progress_cb=cb)
        return {"ok": True, "result": r, "log": buf.getvalue()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "log": buf.getvalue()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root",
                    default="/data5T/liujin/dataset/gm100-cobotmagic-lerobot-demo")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent / "processed_dataset"))
    ap.add_argument("--task", nargs="+", default=None,
                    help="task dir name(s), e.g. task_00001 (default: all)")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF),
                    help="Piper-arm URDF used for eef forward kinematics")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip tasks that already have a .convert_done marker "
                         "(safe resume for large batches)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep converting remaining tasks after a task fails; "
                         "failed tasks are listed at the end (retry with --skip-existing)")
    ap.add_argument("--workers", type=int, default=8,
                    help="tasks converted in parallel (1 = sequential; default: 8)")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable progress/ETA lines")
    ap.add_argument("--progress-interval", type=float, default=10.0,
                    help="seconds between progress lines (default: 10)")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_tasks = args.task or sorted(d.name for d in src_root.iterdir()
                                    if d.is_dir() and d.name.startswith("task_"))

    # --skip-existing: completed tasks are removed from both work and ETA totals
    tasks: list[str] = []
    n_skipped = 0
    for t in all_tasks:
        if args.skip_existing and (out_root / t / ".convert_done").exists():
            log(f"[SKIP] {t}: already converted (.convert_done present)")
            n_skipped += 1
        else:
            tasks.append(t)

    frames_by_task: dict[str, int] = {}
    for t in tasks:
        try:
            frames_by_task[t] = task_frame_count(src_root, t)
        except Exception as e:  # noqa: BLE001  (broken task must not abort the batch)
            log(f"WARN: cannot pre-scan {t}: {e}")
            frames_by_task[t] = 0
    total_frames = sum(frames_by_task.values())
    prog = Progress(total_frames, "frames", "convert",
                    enabled=not args.no_progress,
                    interval=args.progress_interval)
    prog.start_heartbeat()

    log(f"TASK_START total={len(all_tasks)} pending={len(tasks)} "
        f"skipped={n_skipped} frames={total_frames:,}")
    results: list[dict] = []
    failed: list[str] = []
    workers = max(1, int(args.workers))

    if workers == 1:
        for t in tasks:
            r = None
            try:
                r = convert_task(src_root, t, out_root, Path(args.urdf),
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
                futs = {ex.submit(_convert_worker, str(src_root), t,
                                  str(out_root), str(args.urdf), q): t for t in tasks}
                cancelled = False
                for fut in as_completed(futs):
                    t = futs[fut]
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
