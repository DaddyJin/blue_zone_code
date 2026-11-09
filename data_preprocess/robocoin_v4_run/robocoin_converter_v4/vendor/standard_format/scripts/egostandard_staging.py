#!/usr/bin/env python
"""EgoStandard MCAP -> per-task LeRobot v3.0 *staging* for unified_pipeline_v3.

Reuses the audited episode-parsing primitives from egostandard_convert_core.py
(same mcap decoding; video path now transcodes to 960x728, see below) but
aggregates a whole task directory into ONE v3.0 dataset that `batch_robot.py`
can consume directly (detect_format -> "v30", passthrough adapter):

    <out>/<subset>__<task>/
    ├── data/chunk-000/file-000.parquet        # all episodes, fixed_size_list cols
    ├── videos/observation.images.<cam>/chunk-000/file-<ep:03d>.mp4
    └── meta/
        ├── info.json          # v3.0, video features in official format
        ├── modality.json      # 55D slot mapping (world-frame eef + intermediates)
        ├── tasks.parquet      # task_index + task columns
        ├── episodes/chunk-000/file-000.parquet  # full v3 index cols + extras
        ├── subtasks.parquet   # OFFICIAL subtask table (subtask_index -> subtask)
        └── episodes.jsonl     # action_config sidecar (source of truth) + extras

Videos are TRANSCODED during staging: 1920x1456 -> 960x728 (exactly 0.5x,
same aspect), h264 yuv420p. Per-frame camera intrinsics (calibration.*.K/P)
are scaled by the same 0.5 factor so they stay consistent with the pixels.

Subtask annotations use the OFFICIAL LeRobot convention: per-frame
subtask_index column (int64, -1 = unannotated) in the data parquet +
meta/subtasks.parquet (subtask string as pandas index, subtask_index column —
same layout as official tasks.parquet). The agibot-compatible action_config
list is ALSO kept in meta/episodes.jsonl as the human-readable source of
truth (it carries the same boundaries, computed on the real clock).

55D mapping (via modality.json -> canonicalize55):
    left/right_eef     <- wrist (hand joint 0) pos+quat in the per-frame
                          TORSO-YAW frame (gravity-aligned; pelvis origin for
                          EgoStandard-body, head-origin + head-yaw proxy for
                          EgoStandard) — the ego analog of the robot BASE frame:
                          locomotion never leaks into eef coordinates
    left/right_hand    <- NOT mapped (mask=0); full 21-joint poses stay in the
                          passthrough columns (world frame) for later precise
                          retargeting
    left/right_gripper <- pinch distance normalized to [0,1] over [0.04, 0.10] m
    head               <- yaw relative to torso yaw + pitch, from /pose/head
Subtask annotations: official per-frame subtask_index + meta/subtasks.parquet
(primary), plus the agibot-compatible action_config sidecar in
meta/episodes.jsonl (source of truth). Everything else (all 21-joint hand
poses, body poses, per-frame intrinsics/extrinsics, bad-frame flags, source
timestamps) rides along as passthrough columns in the final 55D parquet.

Coordinate frames (verified by probe): all /pose/* topics live in a per-episode
"world" frame (Z-up, anchored near the episode-start head position).
/sensor/camera/head_left/extrinsic is the world->head_left_camera pose per
frame, and /pose/headcam is IDENTICAL to it; likewise /pose/right_eye_cam ==
head_right extrinsic. Cam-frame quantities can be derived on demand from the
passthrough extrinsics + world poses: p_cam = R_wcl^T (p_world - t_wcl).

Usage:
  egostandard_staging.py --tasks "EgoStandard/Adapter Storage" ... \
      [--episodes-per-task 0] [--out /mnt/disk2/egostandard_staging]
Run under: ionice -c3 nice -n 19. Task-monitor generic log format on stdout.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google.protobuf.json_format import MessageToDict
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egostandard_convert_core import (  # noqa: E402
    find_ffmpeg,
    list_channels,
    VIDEO_SCHEMA,
    FPS,
)
from lw_egosuite_backend.mcap_reader import iter_messages  # noqa: E402

SRC_ROOT = Path("/mnt/disk1/EgoStandard")
DST_ROOT = Path("/mnt/disk2/egostandard_staging")

# Video downscale: source 1920x1456 -> 960x728 (exactly 0.5x, same aspect).
# Camera intrinsics K/P are scaled by the same factors (see parse_episode).
SRC_W, SRC_H = 1920, 1456
TGT_W, TGT_H = 960, 728
SCALE_X, SCALE_Y = TGT_W / SRC_W, TGT_H / SRC_H
VIDEO_CRF, VIDEO_PRESET = "18", "fast"
# x264 threads=auto spawns ~1.5x nproc threads PER ffmpeg process; with
# task-level parallelism this thrashes the machine (observed: 8 procs x 48
# threads on 32 cores -> load 121, per-episode throughput down ~3.5x).
# Cap threads and scale task parallelism instead (driver: 5 tasks x 2 cams x 3).
VIDEO_THREADS = "3"


class FfmpegScalePipe:
    """Feed raw annex-b h264 packets into one ffmpeg decode->scale->encode run."""

    def __init__(self, ffmpeg: str, out_path: str):
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "h264",
             "-framerate", str(int(FPS)), "-i", "-",
             "-vf", f"scale={TGT_W}:{TGT_H}", "-c:v", "libx264",
             "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET, "-threads", VIDEO_THREADS,
             "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", out_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.count = 0

    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.count += 1

    def close(self) -> None:
        self.proc.stdin.close()
        _, err = self.proc.communicate(timeout=7200)
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg transcode failed: {err.decode(errors='replace')}")

# Hand joint indices (MediaPipe order), from lw-egosuite-devkit
# converters/pose/std_mcap.py HAND_FRAME_NAMES.
J_WRIST, J_THUMB_TIP, J_INDEX_TIP = 0, 4, 8

PINCH_MIN_M, PINCH_MAX_M = 0.04, 0.10  # EgoDex convention

# Intermediate column names (staging data parquet). modality.json points the
# canonical slots at these (eef maps DIRECTLY onto observation.pose.* wrist
# slices in the per-episode world frame — no intermediate needed).
INTERMEDIATE_STATE = [
    "observation.intermediate.left_wrist_pose_torso",     # (7) xyz + quat wxyz, TORSO frame
    "observation.intermediate.right_wrist_pose_torso",
    "observation.intermediate.left_pinch_norm",           # (1) [0,1]
    "observation.intermediate.right_pinch_norm",
    "observation.intermediate.head_yaw_pitch_torso",      # (2) rad, yaw rel. torso yaw
]
INTERMEDIATE_ACTION = [  # shifted-by-one copies of the state intermediates
    "action.intermediate.left_wrist_pose_torso",
    "action.intermediate.right_wrist_pose_torso",
    "action.intermediate.left_pinch_norm",
    "action.intermediate.right_pinch_norm",
    "action.intermediate.head_yaw_pitch_torso",
]

MODALITY = {
    # eef = wrist (hand joint 0) pos+quat in the per-frame TORSO-YAW frame
    # (gravity-aligned z; origin=pelvis for EgoStandard-body, head otherwise;
    # yaw from shoulder line, or head yaw as proxy when no body joints).
    # This is the ego analog of the robot BASE frame used by agibot/droid/yam:
    # locomotion and world-anchor placement do not leak into eef coordinates.
    "state": {
        "left_eef_position": {"original_key": "observation.intermediate.left_wrist_pose_torso",
                              "start": 0, "end": 3, "absolute": True},
        "left_eef_orientation": {"original_key": "observation.intermediate.left_wrist_pose_torso",
                                 "start": 3, "end": 7, "rotation_type": "wxyz", "absolute": True},
        "right_eef_position": {"original_key": "observation.intermediate.right_wrist_pose_torso",
                               "start": 0, "end": 3, "absolute": True},
        "right_eef_orientation": {"original_key": "observation.intermediate.right_wrist_pose_torso",
                                  "start": 3, "end": 7, "rotation_type": "wxyz", "absolute": True},
        "left_gripper": {"original_key": "observation.intermediate.left_pinch_norm",
                         "start": 0, "end": 1, "absolute": True},
        "right_gripper": {"original_key": "observation.intermediate.right_pinch_norm",
                          "start": 0, "end": 1, "absolute": True},
        "head": {"original_key": "observation.intermediate.head_yaw_pitch_torso",
                 "start": 0, "end": 2, "absolute": True},
    },
    # action = next-frame state (audited convention): shifted copies.
    "action": {
        "left_eef_position": {"original_key": "action.intermediate.left_wrist_pose_torso",
                              "start": 0, "end": 3, "absolute": True},
        "left_eef_orientation": {"original_key": "action.intermediate.left_wrist_pose_torso",
                                 "start": 3, "end": 7, "rotation_type": "wxyz", "absolute": True},
        "right_eef_position": {"original_key": "action.intermediate.right_wrist_pose_torso",
                               "start": 0, "end": 3, "absolute": True},
        "right_eef_orientation": {"original_key": "action.intermediate.right_wrist_pose_torso",
                                  "start": 3, "end": 7, "rotation_type": "wxyz", "absolute": True},
        "left_gripper": {"original_key": "action.intermediate.left_pinch_norm",
                         "start": 0, "end": 1, "absolute": True},
        "right_gripper": {"original_key": "action.intermediate.right_pinch_norm",
                          "start": 0, "end": 1, "absolute": True},
        "head": {"original_key": "action.intermediate.head_yaw_pitch_torso",
                 "start": 0, "end": 2, "absolute": True},
    },
}


def log(line: str) -> None:
    print(line, flush=True)


def sanitize(name: str) -> str:
    return name.replace(" ", "_")


# ---------------------------------------------------------------- episode parse

def parse_episode(mcap_path: str) -> dict:
    """Parse one mcap into in-memory columns (no writes). Raises on defects."""
    channels = list_channels(mcap_path)
    video_topics = sorted(t for t, s in channels.items() if s == VIDEO_SCHEMA)
    other_topics = sorted(t for t in channels if t not in video_topics)
    cam_of = {t: t.split("/")[3] for t in video_topics}  # /sensor/camera/<cam>/video

    per_topic: dict[str, list[tuple[int, object]]] = {t: [] for t in other_topics}
    session_meta: dict | None = None
    semantic_segments: list[dict] = []
    task_description: str | None = None
    for m in iter_messages(mcap_path, topics=other_topics):
        topic, msg = m.topic, m.message
        if topic == "/session/metadata":
            session_meta = MessageToDict(msg, preserving_proto_field_name=True)
            continue
        if topic == "/annotation/semantic_segments":
            if msg.task_description and task_description is None:
                task_description = msg.task_description
            seg = msg.segment
            if seg.subtask_description or seg.end_time:
                semantic_segments.append(
                    {"subtask_description": seg.subtask_description, "end_time": seg.end_time})
            continue
        schema = channels[topic]
        if schema.startswith("pose."):
            per_topic[topic].append((m.log_time_ns, np.array(
                [v for t in msg.transforms
                 for v in (t.pos.x, t.pos.y, t.pos.z, t.quat.w, t.quat.x, t.quat.y, t.quat.z)],
                dtype=np.float64)))
        elif schema == "foxglove.CameraCalibration":
            per_topic[topic].append((m.log_time_ns, {"K": list(msg.K), "R": list(msg.R),
                                                     "P": list(msg.P)}))
        elif schema == "foxglove.FrameTransforms":
            per_topic[topic].append((m.log_time_ns, np.array(
                [v for t in msg.transforms
                 for v in (t.translation.x, t.translation.y, t.translation.z,
                           t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)],
                dtype=np.float64)))
        elif topic.startswith("/annotation/bad_frame/"):
            per_topic[topic].append(
                (m.log_time_ns, (bool(getattr(msg, "is_bad", False)),
                                 [str(x) for x in getattr(msg, "problem_type", [])])))
        else:
            per_topic[topic].append((m.log_time_ns, str(msg)))

    clock = [ts for ts, _ in per_topic["/pose/head"]]
    n_frames = len(clock)
    if n_frames < 2:
        raise RuntimeError(f"too few frames: {n_frames}")
    t_rel0 = clock[0]
    # native capture rate of this episode (the dataset mixes 25fps and 30fps
    # episodes, sometimes within one task). The video is ALWAYS packed at 30fps
    # (LeRobot v3 has a single fps per dataset), so for 25fps-native episodes
    # the container time axis is compressed by 25/30; the real clock is kept in
    # the source_timestamp column and source_fps / source_duration_s meta.
    source_fps = round((n_frames - 1) / ((clock[-1] - t_rel0) / 1e9), 2)
    source_duration_s = (clock[-1] - t_rel0) / 1e9

    def aligned(topic: str) -> list:
        entries = per_topic[topic]
        if len(entries) != n_frames:
            raise RuntimeError(f"{topic}: {len(entries)} msgs vs {n_frames} frames")
        ts_list = [ts for ts, _ in entries]
        if ts_list == clock:
            return [v for _, v in entries]
        out = []
        for t in clock:
            i = bisect.bisect_left(ts_list, t)
            cands = [x for x in (i - 1, i) if 0 <= x < len(entries)]
            out.append(entries[min(cands, key=lambda x: abs(ts_list[x] - t))][1])
        return out

    timestamps = np.array([(ts - t_rel0) / 1e9 for ts in clock], dtype=np.float64)

    # full-fidelity per-frame columns: name -> (per-frame ndarray rows, kind)
    vec_cols: dict[str, np.ndarray] = {}       # float32 fixed-size vectors
    bool_cols: dict[str, list[bool]] = {}
    str_cols: dict[str, list[str]] = {}
    for topic in sorted(per_topic):
        if not per_topic[topic]:
            continue
        schema = channels[topic]
        vals = aligned(topic)
        if schema.startswith("pose."):
            short = topic.removeprefix("/pose/").replace("/", "_")
            vec_cols[f"observation.pose.{short}"] = np.stack(vals).astype(np.float32)
        elif schema == "foxglove.CameraCalibration":
            cam = topic.split("/")[3]
            for mat in ("K", "R", "P"):
                arr = np.stack([v[mat] for v in vals]).astype(np.float32)
                # Intrinsics follow the video downscale (1920x1456 -> 960x728):
                # K row-major [fx,0,cx, 0,fy,cy, 0,0,1]; P row-major 3x4.
                # R (distortion-model rectification) is resolution-independent.
                if mat == "K":
                    arr[:, [0, 2]] *= SCALE_X
                    arr[:, [4, 5]] *= SCALE_Y
                elif mat == "P":
                    arr[:, [0, 2, 3]] *= SCALE_X
                    arr[:, [5, 6, 7]] *= SCALE_Y
                vec_cols[f"observation.calibration.{cam}.{mat}"] = arr
        elif schema == "foxglove.FrameTransforms":
            cam = topic.split("/")[3]
            vec_cols[f"observation.extrinsics.{cam}"] = np.stack(vals).astype(np.float32)
        elif topic.startswith("/annotation/bad_frame/"):
            short = topic.removeprefix("/annotation/bad_frame/").replace("/", "_")
            bool_cols[f"annotation.bad_frame.{short}"] = [v[0] for v in vals]
            str_cols[f"annotation.bad_frame.{short}.problem_types"] = [json.dumps(v[1]) for v in vals]

    # ---- intermediate 55D-mapping columns (per-frame TORSO-YAW frame) --------
    # Head yaw/pitch from /pose/head quat (wxyz), R = Rz(yaw)Ry(pitch)Rx(roll).
    head6 = vec_cols["observation.pose.head"].astype(np.float64)
    R_wh = Rotation.from_quat(head6[:, [4, 5, 6, 3]])
    m = R_wh.as_matrix()
    head_yaw = np.arctan2(m[:, 1, 0], m[:, 0, 0])
    head_pitch = np.arctan2(-m[:, 2, 0], np.hypot(m[:, 1, 0], m[:, 0, 0]))

    # Torso-yaw frame: gravity-aligned (z up), yaw = facing direction.
    #   EgoStandard-body: origin = pelvis, yaw from the shoulder line
    #     (left_shoulder - right_shoulder = body +y; forward = that yaw - 90°);
    #     measured quality: pelvis jitter ~0.9mm, yaw step p95 ~0.7°/frame.
    #   EgoStandard (no /pose/body): origin = head position, yaw = head yaw
    #     (proxy error vs true torso yaw measured on the body subset:
    #      ~5.8° median / ~14° p95; smoothing gives no improvement).
    if "observation.pose.body" in vec_cols:
        body = vec_cols["observation.pose.body"].astype(np.float64).reshape(n_frames, -1, 7)
        torso_origin = body[:, 0, 0:3]                                    # pelvis
        sv = body[:, 16, 0:3] - body[:, 17, 0:3]                          # left_sh - right_sh
        torso_yaw = np.arctan2(sv[:, 1], sv[:, 0]) - np.pi / 2            # facing
        torso_src = "body_joints(pelvis+shoulders)"
    else:
        torso_origin = head6[:, 0:3]
        torso_yaw = head_yaw
        torso_src = "head_yaw_proxy(no /pose/body in source)"

    cy, sy = np.cos(torso_yaw), np.sin(torso_yaw)
    R_torso_inv = Rotation.from_euler("z", -torso_yaw[:, None])  # world -> torso rotation

    def _to_torso(p: np.ndarray) -> np.ndarray:
        d = p - torso_origin
        return np.stack([cy * d[:, 0] + sy * d[:, 1],
                         -sy * d[:, 0] + cy * d[:, 1], d[:, 2]], axis=1)

    inter: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        hand = vec_cols[f"observation.pose.{side}_hand"].astype(np.float64).reshape(n_frames, 21, 7)
        w_pos_t = _to_torso(hand[:, J_WRIST, 0:3])
        r_world = Rotation.from_quat(hand[:, J_WRIST, 3:7][:, [1, 2, 3, 0]])  # wxyz -> xyzw
        q_t_xyzw = (R_torso_inv * r_world).as_quat()
        q_t_wxyz = q_t_xyzw[:, [3, 0, 1, 2]]
        inter[f"observation.intermediate.{side}_wrist_pose_torso"] = np.concatenate(
            [w_pos_t, q_t_wxyz], axis=1).astype(np.float32)
        thumb_p, index_p = hand[:, J_THUMB_TIP, 0:3], hand[:, J_INDEX_TIP, 0:3]
        pinch = np.linalg.norm(thumb_p - index_p, axis=1)  # frame-independent
        inter[f"observation.intermediate.{side}_pinch_norm"] = np.clip(
            (pinch - PINCH_MIN_M) / (PINCH_MAX_M - PINCH_MIN_M), 0.0, 1.0
        )[:, None].astype(np.float32)

    yaw_rel = (head_yaw - torso_yaw + np.pi) % (2 * np.pi) - np.pi
    inter["observation.intermediate.head_yaw_pitch_torso"] = np.stack(
        [yaw_rel, head_pitch], axis=1).astype(np.float32)

    # action = next-frame state (last frame copies itself) — audited convention
    for name in INTERMEDIATE_STATE:
        arr = inter[name]
        inter[name.replace("observation.intermediate.", "action.intermediate.")] = np.concatenate(
            [arr[1:], arr[-1:]], axis=0)

    # ---- action_config (agibot-compatible sidecar) + per-frame subtask ------
    end_times = [s["end_time"] for s in semantic_segments]
    if end_times:
        # Source annotations occasionally have NON-MONOTONIC end_times (e.g. a
        # zero-duration segment whose end_time equals the previous one — observed
        # in 3+ tasks). Clamp to a running max so bounds stay monotonic; the
        # zero-width segment keeps an EMPTY action_config range (frame-exact)
        # and simply never owns a frame.
        for k in range(1, len(end_times)):
            end_times[k] = max(end_times[k], end_times[k - 1])
        # segment k covers frames bounds[k] <= i < bounds[k+1], where the bound
        # is the first frame AFTER the segment's end_time (bisect_right on the
        # real clock). The per-frame column is derived FROM THESE BOUNDS (not
        # from an independent bisect over end_times) so the two are identical
        # by construction — an earlier version bisected end_times per frame,
        # which put the boundary frame (t == end_time) and any frames past the
        # last end_time in the wrong segment (-1), off-by-one vs action_config.
        bounds = [0] + [int(bisect.bisect_right(timestamps, et)) for et in end_times[:-1]] \
                 + [n_frames]
        action_config = [
            {"action_text": semantic_segments[k]["subtask_description"], "skill": None,
             "start_frame": bounds[k], "end_frame": bounds[k + 1]}
            for k in range(len(semantic_segments))
        ]
        subtask_texts = [
            semantic_segments[bisect.bisect_right(bounds, i) - 1]["subtask_description"] or None
            for i in range(n_frames)
        ]
    else:
        action_config = []
        subtask_texts = None

    return {
        "n_frames": n_frames,
        "timestamps": timestamps,
        "source_fps": source_fps,
        "source_duration_s": source_duration_s,
        "vec_cols": vec_cols,
        "bool_cols": bool_cols,
        "str_cols": str_cols,
        "inter": inter,
        "task_description": task_description,
        "semantic_segments": semantic_segments,
        "action_config": action_config,
        "subtask_texts": subtask_texts,  # per-frame subtask text or None
        "torso_src": torso_src,          # how the torso-yaw frame was built
        "session_meta": session_meta,
        "video_topics": video_topics,
        "cam_of": cam_of,
    }


def copy_episode_videos(mcap_path: str, ep: dict, videos_root: Path, ep_idx: int,
                        ffmpeg: str) -> dict[str, int]:
    """Transcode H.264 packets -> 960x728 per-episode mp4 per camera.

    Single ffmpeg pass per camera (decode -> scale -> libx264), written
    directly to the final v3.0 video path. Returns packet counts.
    """
    pipes = {}
    for topic in ep["video_topics"]:
        cam = ep["cam_of"][topic]
        vdir = videos_root / f"observation.images.{cam}" / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        pipes[topic] = FfmpegScalePipe(ffmpeg, str(vdir / f"file-{ep_idx:03d}.mp4"))
    for m in iter_messages(mcap_path, topics=ep["video_topics"]):
        data = getattr(m.message, "data", None)
        if isinstance(data, (bytes, bytearray)):
            pipes[m.topic].write(bytes(data))
    counts = {}
    for topic in ep["video_topics"]:
        pipes[topic].close()
        counts[ep["cam_of"][topic]] = pipes[topic].count
    return counts


# ---------------------------------------------------------------- task assembly

def _fixed_list_f32(arr: np.ndarray) -> pa.Array:
    if arr.shape[1] == 1:
        # 1-dim vectors must be stored as scalars: lerobot maps a shape-[1]
        # float32 feature to a scalar Value, and the HF cast rejects
        # fixed_size_list<float>[1] -> float.
        return pa.array(arr[:, 0], type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(
        pa.array(arr.ravel(), type=pa.float32()), arr.shape[1])


def convert_task(subset: str, task: str, out_root: Path, max_episodes: int,
                 sample_every: int = 1, skip_first: int = 0) -> dict:
    task_dir = SRC_ROOT / subset / "mcap" / task
    mcaps = sorted(f for f in task_dir.iterdir()
                   if f.suffix == ".mcap" and f.stat().st_size > 0
                   and not Path(str(f) + ".aria2").exists())
    if skip_first > 0:
        # continuation of a prefix run: skip exactly the episodes a previous
        # --sample-every run converted (caller's k must match that run's
        # max(1, n//sample_every); verified against episodes.jsonl source_mcap).
        mcaps = mcaps[skip_first:]
    elif sample_every > 1:
        # deterministic CONTIGUOUS PREFIX (e.g. 5 -> first ~20% of episodes per
        # task), so later runs can continue with the remainder; converted
        # episodes are traceable via episodes.jsonl source_mcap.
        k = max(1, len(mcaps) // sample_every)
        mcaps = mcaps[:k]
    if max_episodes > 0:
        mcaps = mcaps[:max_episodes]
    if not mcaps:
        raise RuntimeError(f"no complete mcaps in {task_dir}")

    out_dir = out_root / f"{subset}__{sanitize(task)}"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    ffmpeg = find_ffmpeg()

    episodes: list[dict] = []
    for i, f in enumerate(mcaps):
        t0 = time.perf_counter()
        try:
            ep = parse_episode(str(f))
            ep_idx = len(episodes)  # sequential over successes: video file naming,
            ep["video_packets"] = copy_episode_videos(str(f), ep, out_dir / "videos", ep_idx, ffmpeg)
            # drop episodes whose video track is short/long vs the pose clock
            # (defective source, e.g. Fur_Reversal ep0 right-eye 731/14387):
            # a frame without video is unusable and breaks the 500MB concat
            # frame-count validation downstream.
            bad = {c: (n, ep["n_frames"]) for c, n in ep["video_packets"].items()
                   if n != ep["n_frames"]}
            if bad:
                for cam in ep["cam_of"].values():
                    (out_dir / "videos" / f"observation.images.{cam}" / "chunk-000"
                     / f"file-{ep_idx:03d}.mp4").unlink(missing_ok=True)
                raise RuntimeError(f"video packet count != pose frames: {bad}")
            ep["source_mcap"] = f.name  # data episode_index and episodes meta must agree
            episodes.append(ep)
            log(f"  [{i + 1}/{len(mcaps)}] {f.stem} OK {time.perf_counter() - t0:.1f}s "
                f"frames={ep['n_frames']}")
        except Exception as e:  # noqa: BLE001 - log and skip defective episodes
            log(f"  [{i + 1}/{len(mcaps)}] {f.stem} FAIL {time.perf_counter() - t0:.1f}s {e}")
    if not episodes:
        raise RuntimeError(f"all episodes failed in {task_dir}")

    # task_index per unique task_description (fallback: task dir name)
    task_texts = []
    for ep in episodes:
        t = ep["task_description"] or task
        if t not in task_texts:
            task_texts.append(t)
    task_index_of = {t: i for i, t in enumerate(task_texts)}

    # official LeRobot subtask vocab: distinct subtask descriptions across the
    # task's episodes, first-seen order; -1 marks unannotated frames. Sources:
    # per-frame texts PLUS every action_config text — action_config is the
    # source of truth, and zero-width segments (non-monotonic source end_times,
    # clamped) never own a frame but must still exist in the vocab.
    subtask_texts: list[str] = []
    for ep in episodes:
        for t in ep["subtask_texts"] or []:
            if t is not None and t not in subtask_texts:
                subtask_texts.append(t)
        for seg in ep["action_config"]:
            t = seg["action_text"] or None
            if t is not None and t not in subtask_texts:
                subtask_texts.append(t)
    subtask_index_of = {t: i for i, t in enumerate(subtask_texts)}

    # ---- data parquet ---------------------------------------------------------
    vec_names = sorted({n for ep in episodes for n in ep["vec_cols"]})
    bool_names = sorted({n for ep in episodes for n in ep["bool_cols"]})
    str_names = sorted({n for ep in episodes for n in ep["str_cols"]})
    inter_names = INTERMEDIATE_STATE + INTERMEDIATE_ACTION
    for n in inter_names:  # every episode must provide all intermediate cols
        assert all(n in ep["inter"] for ep in episodes), n

    cols: dict[str, pa.Array] = {}
    cols["episode_index"] = pa.array(
        [i for i, ep in enumerate(episodes) for _ in range(ep["n_frames"])], type=pa.int64())
    cols["frame_index"] = pa.array(
        [j for ep in episodes for j in range(ep["n_frames"])], type=pa.int64())
    cols["task_index"] = pa.array(
        [task_index_of[ep["task_description"] or task]
         for ep in episodes for _ in range(ep["n_frames"])], type=pa.int64())
    cols["timestamp"] = pa.array(
        np.concatenate([np.arange(ep["n_frames"]) / FPS for ep in episodes]), type=pa.float64())
    cols["source_timestamp"] = pa.array(
        np.concatenate([ep["timestamps"] for ep in episodes]), type=pa.float64())
    # official per-frame subtask column (paired with meta/subtasks.parquet);
    # boundaries identical to episodes.jsonl action_config (real-clock bisect)
    cols["subtask_index"] = pa.array(
        [subtask_index_of[t] if t is not None else -1
         for ep in episodes
         for t in (ep["subtask_texts"] or [None] * ep["n_frames"])],
        type=pa.int64())
    for n in vec_names:
        cols[n] = _fixed_list_f32(np.concatenate([ep["vec_cols"][n] for ep in episodes]))
    for n in inter_names:
        cols[n] = _fixed_list_f32(np.concatenate([ep["inter"][n] for ep in episodes]))
    for n in bool_names:
        cols[n] = pa.array([v for ep in episodes for v in ep["bool_cols"][n]], type=pa.bool_())
    for n in str_names:
        cols[n] = pa.array([v for ep in episodes for v in ep["str_cols"][n]], type=pa.string())

    data_dir = out_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), data_dir / "file-000.parquet", compression="zstd")

    # ---- meta -------------------------------------------------------------------
    meta_dir = out_dir / "meta"
    (meta_dir / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    cams = sorted({c for ep in episodes for c in ep["cam_of"].values()})
    frame_cursor = 0
    ep_rows = []
    ep_jsonl = []
    for i, ep in enumerate(episodes):
        n = ep["n_frames"]
        t_text = ep["task_description"] or task
        row = {
            "episode_index": i,
            "tasks": [t_text],
            "length": n,
            "task_index": task_index_of[t_text],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": frame_cursor,
            "dataset_to_index": frame_cursor + n,
            "semantic_segments": json.dumps(ep["semantic_segments"], ensure_ascii=False),
            "session_metadata": json.dumps(ep["session_meta"] or {}, ensure_ascii=False),
            "source_fps": ep["source_fps"],
            "source_duration_s": ep["source_duration_s"],
        }
        for cam in cams:
            row[f"videos/observation.images.{cam}/chunk_index"] = 0
            row[f"videos/observation.images.{cam}/file_index"] = i
            row[f"videos/observation.images.{cam}/from_timestamp"] = 0.0
            row[f"videos/observation.images.{cam}/to_timestamp"] = n / FPS
        ep_rows.append(row)
        ep_jsonl.append({
            # agibot episodes.jsonl format: episode_index/tasks/length/action_config
            "episode_index": i,
            "tasks": [t_text],
            "length": n,
            "action_config": ep["action_config"],
            # ego full-fidelity extras (consumed by downstream tooling, ignored
            # by agibot-compatible readers)
            "semantic_segments": ep["semantic_segments"],
            "session_metadata": ep["session_meta"],
            "source_mcap": ep["source_mcap"],
            "source_fps": ep["source_fps"],
            "source_duration_s": ep["source_duration_s"],
            "torso_frame_src": ep["torso_src"],
        })
        frame_cursor += n
    pq.write_table(
        pa.Table.from_pylist(ep_rows),
        meta_dir / "episodes" / "chunk-000" / "file-000.parquet", compression="zstd")
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in ep_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pq.write_table(
        pa.Table.from_pydict({"task_index": list(range(len(task_texts))), "task": task_texts}),
        meta_dir / "tasks.parquet")

    # meta/subtasks.parquet in the OFFICIAL layout: subtask string as the pandas
    # index + subtask_index column (same as official tasks.parquet). forge_venv
    # has no pandas, so the pandas index metadata is attached by hand — the
    # audit round-trips it with pd.read_parquet in the lerobot env.
    st_tbl = pa.table({"subtask_index": pa.array(range(len(subtask_texts)), type=pa.int64()),
                       "subtask": pa.array(subtask_texts, type=pa.string())})
    pandas_meta = {
        "index_columns": ["subtask"],
        "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode",
                            "numpy_type": "object", "metadata": None}],
        "columns": [
            {"name": "subtask_index", "field_name": "subtask_index", "pandas_type": "int64",
             "numpy_type": "int64", "metadata": None},
            {"name": "subtask", "field_name": "subtask", "pandas_type": "unicode",
             "numpy_type": "object", "metadata": None},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    st_tbl = st_tbl.replace_schema_metadata(
        {**(st_tbl.schema.metadata or {}), b"pandas": json.dumps(pandas_meta).encode()})
    pq.write_table(st_tbl, meta_dir / "subtasks.parquet")
    (meta_dir / "modality.json").write_text(json.dumps(MODALITY, indent=2, ensure_ascii=False))

    features: dict[str, dict] = {}
    for n, arr in cols.items():
        t = arr.type
        if pa.types.is_fixed_size_list(t):
            features[n] = {"dtype": "float32", "shape": [t.list_size], "names": None}
        elif t == pa.float32():
            features[n] = {"dtype": "float32", "shape": [1], "names": None}
        elif t == pa.bool_():
            features[n] = {"dtype": "bool", "shape": [1], "names": None}
        elif t == pa.string():
            features[n] = {"dtype": "string", "shape": [1], "names": None}
        elif t == pa.int64():
            features[n] = {"dtype": "int64", "shape": [1], "names": None}
        else:
            features[n] = {"dtype": "float64", "shape": [1], "names": None}
    for cam in cams:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [TGT_H, TGT_W, 3], "names": ["height", "width", "rgb"],
            "info": {"video.height": TGT_H, "video.width": TGT_W, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "video.fps": int(FPS), "video.channels": 3, "has_audio": False},
        }
    total_frames = sum(ep["n_frames"] for ep in episodes)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "human_ego",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task_texts),
        "chunks_size": 1000,
        "fps": int(FPS),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "custom": {
            "converter": "egostandard_staging.py (per-task staging for unified_pipeline_v3)",
            "source": f"{subset}/mcap/{task}",
            "coordinate_frames": {
                "torso": "PER-FRAME torso-yaw frame for ALL 55D active slots (eef/head): "
                         "gravity-aligned z, yaw = facing direction, origin = pelvis for "
                         "EgoStandard-body (shoulder-line yaw), origin = head with head-yaw "
                         "proxy for EgoStandard (no /pose/body; measured proxy error "
                         "~5.8deg median / ~14deg p95, no systematic bias). This is the ego "
                         "analog of the robot BASE frame (agibot/droid/yam): locomotion and "
                         "world-anchor placement do not leak into eef coordinates. "
                         "Per-episode source: episodes.jsonl torso_frame_src.",
                "world": "per-episode world frame, Z-up; ALL observation.pose.* passthrough "
                         "columns stay in this frame (full fidelity — any other frame can be "
                         "derived on demand). NOTE: a few source episodes anchor the world "
                         "origin far from the subject (observed up to ~76m); this never "
                         "affects the 55D slots, which are torso-relative.",
                "cam": "head_left camera frame == /pose/headcam == head_left extrinsic; "
                       "derive cam-frame quantities on demand as p_cam = R_wcl^T (p_world - t_wcl) "
                       "from the passthrough extrinsics + world poses",
                "quat_orders": "observation.pose.*: wxyz (proto order); "
                               "observation.extrinsics.*: xyzw; intermediate wrist poses: wxyz; "
                               "state_unified eef quats: xyzw",
            },
            "hand_joint_order": ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                                 "index_finger_mcp", "index_finger_pip", "index_finger_dip",
                                 "index_finger_tip", "middle_finger_mcp", "middle_finger_pip",
                                 "middle_finger_dip", "middle_finger_tip", "ring_finger_mcp",
                                 "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
                                 "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"],
            "body_joint_order": ["pelvis", "left_hip", "right_hip", "spine1", "left_knee",
                                 "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
                                 "left_foot", "right_foot", "neck", "left_collar", "right_collar",
                                 "head", "left_shoulder", "right_shoulder", "left_elbow",
                                 "right_elbow", "left_wrist_dup", "right_wrist_dup"],
            "body_joint_order_note": "observation.pose.body has 22 entries; the last two are "
                                     "wrist placeholders (duplicate of hand wrist, unused by "
                                     "the body TF per devkit std_mcap.py BODY_FRAME_NAMES)",
            "mapping_55d": {
                "left/right_eef": "wrist (hand joint 0) pos+quat xyzw, per-frame TORSO-YAW "
                                  "frame (see coordinate_frames.torso); via intermediate "
                                  "observation.intermediate.*_wrist_pose_torso (wxyz)",
                "left/right_hand": "NOT mapped (mask=0) — full 21-joint poses kept in "
                                   "observation.pose.*_hand (world frame) for later "
                                   "precise retargeting",
                "left/right_gripper": f"pinch dist normalized over [{PINCH_MIN_M},{PINCH_MAX_M}] m, clipped [0,1]",
                "head": "yaw RELATIVE to torso yaw + gravity-referenced pitch, from /pose/head",
                "action": "next-frame state (last frame copies itself)",
                "subtasks": "OFFICIAL LeRobot convention: per-frame subtask_index column "
                            "(int64, -1 = unannotated) + meta/subtasks.parquet; the "
                            "agibot-compatible action_config list in episodes.jsonl is kept "
                            "as the human-readable source of truth (identical boundaries, "
                            "computed on the real clock)",
            },
            "video_resize": {
                "source": [SRC_H, SRC_W],
                "output": [TGT_H, TGT_W],
                "scale": [SCALE_Y, SCALE_X],
                "encode": f"libx264 crf={VIDEO_CRF} preset={VIDEO_PRESET} yuv420p",
                "calibration": "observation.calibration.*.K/P are PRE-SCALED to the 960x728 "
                               "image plane (fx,fy,cx,cy and the P translation column); "
                               "R and distortion are resolution-independent",
            },
            "time_axis": {
                "container_fps": int(FPS),
                "note": "LeRobot v3 has one fps per dataset; ALL videos are packed at "
                        f"{int(FPS)}fps. The source dataset mixes native 25fps/30fps episodes: "
                        "for 25fps-native episodes the container time axis (timestamp column, "
                        "videos to_timestamp) is compressed by 25/30. Frame indices and "
                        "action_config frame ranges are always exact. Real wall-clock time: "
                        "use the source_timestamp column and per-episode source_fps / "
                        "source_duration_s (episodes parquet + episodes.jsonl).",
                "episodes_by_source_fps": {str(k): v for k, v in sorted(
                    __import__("collections").Counter(ep["source_fps"] for ep in episodes).items())},
            },
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return {"out": str(out_dir), "episodes": len(episodes), "frames": total_frames,
            "cams": cams, "tasks": task_texts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, nargs="+",
                    help="'<subset>/<task name>' e.g. 'EgoStandard/Adapter Storage'")
    ap.add_argument("--episodes-per-task", type=int, default=0, help="0 = all complete mcaps")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="keep the FIRST ~1/N of complete mcaps per task "
                         "(contiguous prefix, resumable; 5 = ~20%%)")
    ap.add_argument("--skip-first", type=int, default=0,
                    help="skip the first N complete mcaps per task (continuation "
                         "of a previous --sample-every prefix run; overrides "
                         "--sample-every when > 0)")
    ap.add_argument("--out", default=str(DST_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    log(f"TASK_START total={len(args.tasks)}")
    results = []
    for i, spec in enumerate(args.tasks):
        subset, task = spec.split("/", 1)
        t0 = time.perf_counter()
        try:
            r = convert_task(subset, task, out_root, args.episodes_per_task,
                             args.sample_every, args.skip_first)
            results.append(r)
            log(f"[{i + 1}/{len(args.tasks)}] {sanitize(subset)}__{sanitize(task)} OK "
                f"{time.perf_counter() - t0:.1f}s eps={r['episodes']} frames={r['frames']}")
        except Exception as e:  # noqa: BLE001
            log(f"[{i + 1}/{len(args.tasks)}] {sanitize(subset)}__{sanitize(task)} FAIL "
                f"{time.perf_counter() - t0:.1f}s {e}")
            raise
    log("TASK_DONE")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
