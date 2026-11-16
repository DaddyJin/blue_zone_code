#!/usr/bin/env python3
"""Build paired A1/B1 RoboCOIN 55D datasets.

A1 is a subset of the already processed full dataset A (data-cleaning output).
B1 has the same episodes/rows/videos, but EEF slots 14:28 are replaced by
URDF-root forward kinematics from the 55D arm joints.

Usage:
    python make_a1_b1.py --input ./A --output ./ablation

Optional (defaults are relative to this script):
    --urdf-root ./urdf
    --calibration ./urdf/validation/calibration.json

Only numpy and pyarrow are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ab_fk import Urdf, pack_pose, rigid_fit
from ab_progress import Progress

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("./A")
DEFAULT_OUTPUT = Path("./ablation")
DEFAULT_URDF_ROOT = SCRIPT_DIR / "urdf"
DEFAULT_CALIBRATION = DEFAULT_URDF_ROOT / "validation" / "calibration.json"

N_DIM = 55
LEFT_ARM_START = 0
RIGHT_ARM_START = 7
LEFT_EEF = (14, 21)
RIGHT_EEF = (21, 28)
EEF_SLICE = slice(14, 28)

TIER1 = {
    "Leju_Kuavo_4 (Pro/two-finger)",
    "leju_robot (LB/five-finger)",
    "Agilex_Cobot_Magic",
    "Unitree_G1",
}
TIER2 = {
    "Airbot_MMK2",
    "Galaxea_R1_Lite (6DoF)",
    "Realman_RMC-AIDA-L",
    "AgiBot_G1",
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fixed_list_to_numpy(arr: pa.ChunkedArray | pa.Array, width: int = N_DIM) -> np.ndarray:
    """FixedSizeList<float>[width] -> (N,width) float32 (writable copy)."""
    try:
        flat = arr.combine_chunks().flatten()
        out = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, width)
        return np.array(out, dtype=np.float32, copy=True)
    except Exception:
        return np.array(arr.to_pylist(), dtype=np.float32, copy=True)


def variable_list_to_numpy(arr: pa.ChunkedArray | pa.Array) -> np.ndarray:
    return np.asarray(arr.to_pylist(), dtype=np.float64)


def fixed_list_array(values: np.ndarray, width: int = N_DIM) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def set_fixed_list(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(f"column {name!r} not found")
    return table.set_column(idx, name, fixed_list_array(values))


def quat_xyzw_to_matrices(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = q.shape[0]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """Angle of (N,3,3) rotations, in degrees."""
    trace = np.trace(R, axis1=1, axis2=2)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# ---------------------------------------------------------------------------
# dataset discovery / family resolution
# ---------------------------------------------------------------------------
def find_datasets(input_root: Path) -> List[Path]:
    if (input_root / "meta" / "info.json").is_file():
        return [input_root]
    found = []
    if input_root.is_dir():
        for child in sorted(input_root.iterdir()):
            if child.is_dir() and (child / "meta" / "info.json").is_file():
                found.append(child)
    if not found:
        raise FileNotFoundError(
            f"no dataset found at {input_root} or its immediate children "
            "(expected */meta/info.json)"
        )
    return found


def _feature_names(info: Mapping[str, Any]) -> Optional[List[str]]:
    features = info.get("features", {})
    for key in ("source.observation.state", "observation.state"):
        f = features.get(key)
        if isinstance(f, dict) and isinstance(f.get("names"), list):
            return list(f["names"])
    return None


def detect_dof(info: Mapping[str, Any], input_manifest: Mapping[str, Any]) -> Tuple[int, int]:
    """Return (left arm DoF, right arm DoF) from source metadata."""
    for fields in input_manifest.get("state_fields", []) or []:
        slot = fields.get("slot")
        if slot == "left_arm_joint":
            return len(fields.get("indices", [])), _right_dof(input_manifest)
    names = _feature_names(info)
    if names:
        left = sum(1 for n in names if re.fullmatch(r"left_arm_joint_\d+_rad", str(n)))
        right = sum(1 for n in names if re.fullmatch(r"right_arm_joint_\d+_rad", str(n)))
        if left or right:
            return left, right
    return 0, 0


def _right_dof(input_manifest: Mapping[str, Any]) -> int:
    for fields in input_manifest.get("state_fields", []) or []:
        if fields.get("slot") == "right_arm_joint":
            return len(fields.get("indices", []))
    return 0


def resolve_family(
    dataset_name: str,
    info: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    n_left: int,
    n_right: int,
) -> Tuple[Optional[str], Optional[str], str]:
    """Return (calibration family, exclusion reason, matched text)."""
    text = " ".join(
        [
            dataset_name,
            str(info.get("robot_type", "")),
            str(input_manifest.get("robot_type", "")),
            str(input_manifest.get("source", {}).get("id", "")),
        ]
    ).lower()

    # Order matters: AgiBot / Galbot / AI2 all contain "g1"/"bot" substrings.
    if "agibot" in text:
        return "AgiBot_G1", None, text
    if "galbot" in text:
        return None, "galbot_no_public_urdf", text
    if "tianq" in text or "tianqing" in text:
        return None, "tianqing_no_public_urdf", text
    if "ai2" in text or "alphabot" in text or "alpha_bot" in text:
        return None, "ai2_public_urdf_mismatch_strict", text

    if "galaxea" in text or "r1_lite" in text or "r1 lite" in text:
        if n_left == 6 and n_right == 6:
            return "Galaxea_R1_Lite (6DoF)", None, text
        if n_left == 7 and n_right == 7:
            return None, "galaxea_7dof_no_7dof_urdf", text
        return None, "galaxea_dof_unknown", text

    if "unitree" in text or "g1edu" in text:
        return "Unitree_G1", None, text

    if "cobot" in text or "split_aloha" in text or "aloha" in text:
        return "Agilex_Cobot_Magic", None, text

    if "kuavo" in text or "leju_robot" in text:
        if "leju_robot" in text or "five" in text or "lb" in text:
            return "leju_robot (LB/five-finger)", None, text
        return "Leju_Kuavo_4 (Pro/two-finger)", None, text

    if "airbot" in text:
        return "Airbot_MMK2", None, text

    if "realman" in text or "rmc-aida" in text or "rmc_aida" in text:
        return "Realman_RMC-AIDA-L", None, text

    return None, "unknown_family", text


def tier_for(family: str) -> Optional[int]:
    if family in TIER1:
        return 1
    if family in TIER2:
        return 2
    return None


# ---------------------------------------------------------------------------
# LeRobot v3.0 metadata / data helpers
# ---------------------------------------------------------------------------
def load_episodes_table(dataset_dir: Path) -> pa.Table:
    ep_dir = dataset_dir / "meta" / "episodes"
    files = sorted(ep_dir.rglob("*.parquet")) if ep_dir.is_dir() else []
    if not files:
        raise RuntimeError(
            f"{dataset_dir}: only LeRobot v3.0 meta/episodes/*.parquet is supported"
        )
    tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SourceReader:
    """Read and cache source data files; slice by episode offsets."""

    def __init__(self, dataset_dir: Path, info: Mapping[str, Any]):
        self.dataset_dir = dataset_dir
        self.info = info
        self.cache: Dict[Path, pa.Table] = {}
        self.template = str(info["data_path"])
        self.chunks_size = int(info.get("chunks_size", 1000))

    def path_for(self, ep: Mapping[str, Any]) -> Path:
        old_ep = int(ep["episode_index"])
        chunk = ep.get("data/chunk_index")
        file_index = ep.get("data/file_index")
        chunk = int(chunk) if chunk is not None else old_ep // self.chunks_size
        file_index = (
            int(file_index) if file_index is not None else old_ep % self.chunks_size
        )
        try:
            rel = self.template.format(
                chunk_index=chunk,
                file_index=file_index,
                episode_chunk=old_ep // self.chunks_size,
                episode_index=old_ep,
            )
        except KeyError as exc:
            raise KeyError(
                f"cannot format data_path={self.template!r} for episode {old_ep}; "
                f"missing template key {exc}"
            ) from exc
        return self.dataset_dir / rel

    def read_episode(self, ep: Mapping[str, Any]) -> pa.Table:
        path = self.path_for(ep)
        if path not in self.cache:
            # Keep at most two source files in memory; episodes are normally
            # grouped by source file.
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = pq.read_table(path)
        table = self.cache[path]
        length = int(ep["length"])
        old_ep = int(ep["episode_index"])

        # v3.0 chunked-file metadata has file_from_offset/file_to_offset.
        # Some data-cleaning outputs omit those fields because each episode is
        # its own file, or because the episode is located by episode_index.
        if "data/file_from_offset" in ep and "data/file_to_offset" in ep:
            start = int(ep["data/file_from_offset"])
            end = int(ep["data/file_to_offset"])
            if end - start != length:
                raise ValueError(
                    f"episode {old_ep}: offset length {end-start} != {length}"
                )
            if end > len(table):
                raise ValueError(f"episode {old_ep}: offset beyond {path}")
            return table.slice(start, length)

        if "episode_index" in table.column_names:
            field_type = table.schema.field("episode_index").type
            mask = pc.equal(table["episode_index"], pa.scalar(old_ep, type=field_type))
            filtered = table.filter(mask)
            if len(filtered) == length:
                return filtered
            if len(filtered) != 0:
                raise ValueError(
                    f"episode {old_ep}: episode_index filter returned "
                    f"{len(filtered)} rows, expected {length} ({path})"
                )

        if len(table) == length:
            return table

        raise ValueError(
            f"episode {old_ep}: cannot locate episode in {path}; "
            f"file rows={len(table)} expected={length}; "
            f"metadata keys={sorted(ep.keys())}"
        )


@dataclass
class EpisodeResult:
    old_episode: int
    new_episode: int
    length: int
    original_row: Dict[str, Any]
    meta_row: Dict[str, Any]
    a1_table: Optional[pa.Table] = None
    b1_table: Optional[pa.Table] = None
    quality: Dict[str, Any] = field(default_factory=dict)


class PairedBufferedWriter:
    """Write A1 and B1 data files with identical file boundaries."""

    def __init__(
        self,
        out_a: Path,
        out_b: Path,
        info: Mapping[str, Any],
        target_bytes: Optional[int] = None,
        one_file_per_episode: bool = False,
    ):
        self.out_a = out_a
        self.out_b = out_b
        self.info = info
        self.chunks_size = int(info.get("chunks_size", 1000))
        self.target_bytes = int(
            target_bytes
            if target_bytes is not None
            else float(info.get("data_files_size_in_mb", 100)) * 1_000_000
        )
        self.one_file_per_episode = bool(one_file_per_episode)
        self.a_buf: List[pa.Table] = []
        self.b_buf: List[pa.Table] = []
        self.rows = 0
        self.bytes = 0
        self.file_index = 0

    def add(self, result: EpisodeResult) -> None:
        if result.a1_table is None or result.b1_table is None:
            raise ValueError("result has no tables")
        nbytes = int(result.a1_table.nbytes)

        if self.one_file_per_episode:
            if self.a_buf:
                self.flush()
            chunk = self.file_index // self.chunks_size
            file_in_chunk = self.file_index % self.chunks_size
            result.meta_row["data/chunk_index"] = chunk
            result.meta_row["data/file_index"] = file_in_chunk
            result.meta_row["data/file_from_offset"] = 0
            result.meta_row["data/file_to_offset"] = result.length
            self.a_buf.append(result.a1_table)
            self.b_buf.append(result.b1_table)
            self.rows = result.length
            self.bytes = nbytes
            self.flush()
            return

        if self.a_buf and self.bytes + nbytes > self.target_bytes:
            self.flush()

        chunk = self.file_index // self.chunks_size
        file_in_chunk = self.file_index % self.chunks_size
        offset = self.rows
        result.meta_row["data/chunk_index"] = chunk
        result.meta_row["data/file_index"] = file_in_chunk
        result.meta_row["data/file_from_offset"] = offset
        result.meta_row["data/file_to_offset"] = offset + result.length
        self.a_buf.append(result.a1_table)
        self.b_buf.append(result.b1_table)
        self.rows += result.length
        self.bytes += nbytes

    def _target(self, root: Path) -> Path:
        chunk = self.file_index // self.chunks_size
        file_in_chunk = self.file_index % self.chunks_size
        rel = str(self.info["data_path"]).format(
            chunk_index=chunk,
            file_index=file_in_chunk,
            episode_chunk=chunk,
            episode_index=self.file_index,
        )
        return root / rel

    def flush(self) -> None:
        if not self.a_buf:
            return
        a_path = self._target(self.out_a)
        b_path = self._target(self.out_b)
        a_path.parent.mkdir(parents=True, exist_ok=True)
        b_path.parent.mkdir(parents=True, exist_ok=True)
        a_table = pa.concat_tables(self.a_buf)
        b_table = pa.concat_tables(self.b_buf)
        tmp_a = a_path.with_suffix(a_path.suffix + ".tmp")
        tmp_b = b_path.with_suffix(b_path.suffix + ".tmp")
        try:
            pq.write_table(a_table, tmp_a, compression="zstd")
            pq.write_table(b_table, tmp_b, compression="zstd")
            os.replace(tmp_a, a_path)
            os.replace(tmp_b, b_path)
        finally:
            for p in (tmp_a, tmp_b):
                if p.exists():
                    p.unlink(missing_ok=True)
        self.file_index += 1
        self.a_buf.clear()
        self.b_buf.clear()
        self.rows = 0
        self.bytes = 0

    def close(self) -> None:
        self.flush()

    @property
    def total_chunks(self) -> int:
        return 0 if self.file_index == 0 else (
            (self.file_index - 1) // self.chunks_size + 1
        )


# ---------------------------------------------------------------------------
# validation / construction
# ---------------------------------------------------------------------------
def validate_config(cfg: Mapping[str, Any], urdf: Urdf, n_left: int, n_right: int) -> None:
    for side, n in (("left", n_left), ("right", n_right)):
        names = list(cfg["arm_joints"][side])
        if len(names) != n:
            raise ValueError(
                f"{side} arm: calibration has {len(names)} joints, data has {n}"
            )
        for name in names:
            if name not in urdf.joints:
                raise KeyError(f"arm joint {name!r} missing from {urdf.path}")
        eef_link = cfg["eef_link"][side]
        urdf.chain(eef_link)  # validates existence
    for name in (cfg.get("extra_joint_mapping") or {}):
        if name not in urdf.joints:
            raise KeyError(f"extra joint {name!r} missing from {urdf.path}")


def reindex_table(table: pa.Table, new_episode: int, new_index_start: int) -> pa.Table:
    n = len(table)
    out = table
    for name, values in (
        ("episode_index", np.full(n, new_episode, dtype=np.int64)),
        ("index", np.arange(new_index_start, new_index_start + n, dtype=np.int64)),
    ):
        idx = out.schema.get_field_index(name)
        if idx < 0:
            raise KeyError(f"required column {name!r} not found")
        out = out.set_column(idx, name, pa.array(values, type=out.schema.field(idx).type))
    return out


def _raw_column(table: pa.Table, kind: str) -> pa.Array:
    candidates = (
        ("source.observation.state", "observation.state")
        if kind == "state"
        else ("source.action", "action")
    )
    for name in candidates:
        if name in table.column_names:
            return table[name]
    raise KeyError(f"none of {candidates} found for extra joint mapping")


def build_b1_table(
    table: pa.Table,
    cfg: Mapping[str, Any],
    urdf: Urdf,
    n_left: int,
    n_right: int,
) -> Tuple[pa.Table, Dict[str, Any]]:
    """Replace EEF 14:28 and its masks with URDF-root FK.

    Arms are taken from 55D slots 0:7 / 7:14. Extra joints (AgiBot body joints)
    are taken from the raw source columns at the same row, because the 55D
    mapping intentionally leaves 42:55 zero.
    """
    unified = {
        "state": fixed_list_to_numpy(table["state_unified"]),
        "action": fixed_list_to_numpy(table["action_unified"]),
    }
    masks = {
        "state": fixed_list_to_numpy(table["mask_state"]),
        "action": fixed_list_to_numpy(table["mask_action"]),
    }
    extra = cfg.get("extra_joint_mapping") or {}
    raw = {}
    if extra:
        for kind in ("state", "action"):
            raw[kind] = variable_list_to_numpy(_raw_column(table, kind))

    stats: Dict[str, Any] = {"sides": {}}
    out = table
    for kind in ("state", "action"):
        arr = unified[kind]
        mask = masks[kind]
        out_arr = arr.copy()
        out_mask = mask.copy()

        for side, start, eef0, eef1 in (
            ("left", LEFT_ARM_START, LEFT_EEF[0], LEFT_EEF[1]),
            ("right", RIGHT_ARM_START, RIGHT_EEF[0], RIGHT_EEF[1]),
        ):
            names = list(cfg["arm_joints"][side])
            n = len(names)
            q = arr[:, start : start + n]
            valid = mask[:, start : start + n].all(axis=1) & np.isfinite(q).all(axis=1)
            qmap: Dict[str, np.ndarray] = {
                name: q[:, i].astype(np.float64) for i, name in enumerate(names)
            }
            if extra:
                raw_arr = raw[kind]
                for name, idx in extra.items():
                    idx = int(idx)
                    if idx >= raw_arr.shape[1]:
                        raise ValueError(
                            f"extra joint {name!r} index {idx} beyond raw width "
                            f"{raw_arr.shape[1]}"
                        )
                    values = raw_arr[:, idx]
                    qmap[name] = values
                    valid &= np.isfinite(values)

            T, root = urdf.fk_batch(
                qmap, cfg["eef_link"][side], n=len(out_arr)
            )
            pos = T[:, :3, 3]
            _, quat = pack_pose(T)
            ok = (
                valid
                & np.isfinite(pos).all(axis=1)
                & np.isfinite(quat).all(axis=1)
            )
            out_arr[ok, eef0 : eef0 + 3] = pos[ok]
            out_arr[ok, eef0 + 3 : eef1] = quat[ok]
            out_arr[~ok, eef0:eef1] = 0.0
            out_mask[:, eef0:eef1] = ok[:, None].astype(np.float32)

            stats["sides"][f"{kind}_{side}"] = {
                "valid_frames": int(ok.sum()),
                "total_frames": int(len(ok)),
                "root_link": root,
                "eef_link": cfg["eef_link"][side],
            }

        out = set_fixed_list(out, f"{kind}_unified", out_arr)
        out = set_fixed_list(out, f"mask_{kind}", out_mask)
        if kind == "state" and "reliability" in out.column_names:
            rel = fixed_list_to_numpy(out["reliability"])
            rel[:, EEF_SLICE] = out_mask[:, EEF_SLICE]
            out = set_fixed_list(out, "reliability", rel)

    return out, stats


def validate_pair(a1: pa.Table, b1: pa.Table, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Check structural identity and measure A1/B1 EEF relation.

    Structural mismatches raise. Quality residuals are reported, not raised:
    Tier 2 families have known model/zero-offset limitations.

    Two residuals are reported for every side:
      * ``fit``: rigid fit of B1 -> A1, i.e. the best possible R,t for this
        dataset. This isolates URDF/model error from frame offset.
      * ``cal``: using the family-level R,t from calibration.json. If A's
        eef_sim frame differs from the calibration sample, this residual is
        large while ``fit`` may still be small.
    """
    if len(a1) != len(b1):
        raise AssertionError(f"row count mismatch: {len(a1)} vs {len(b1)}")

    modified = {
        "state_unified",
        "action_unified",
        "mask_state",
        "mask_action",
        "reliability",
    }
    for name in a1.column_names:
        if name in modified:
            continue
        if not a1[name].equals(b1[name]):
            raise AssertionError(f"column {name!r} differs between A1 and B1")

    result: Dict[str, Any] = {"sides": {}}
    for kind in ("state", "action"):
        x = fixed_list_to_numpy(a1[f"{kind}_unified"])
        y = fixed_list_to_numpy(b1[f"{kind}_unified"])
        mx = fixed_list_to_numpy(a1[f"mask_{kind}"])
        my = fixed_list_to_numpy(b1[f"mask_{kind}"])

        for sl in (slice(0, 14), slice(28, 55)):
            np.testing.assert_array_equal(x[:, sl], y[:, sl])
            np.testing.assert_array_equal(mx[:, sl], my[:, sl])

        for side, eef0, eef1, sidekey in (
            ("left", LEFT_EEF[0], LEFT_EEF[1], "L"),
            ("right", RIGHT_EEF[0], RIGHT_EEF[1], "R"),
        ):
            side_cfg = cfg.get("sides", {}).get(sidekey)
            if not side_cfg:
                continue
            pos_valid = (
                mx[:, eef0 : eef0 + 3].all(axis=1)
                & my[:, eef0 : eef0 + 3].all(axis=1)
            )
            rot_valid = (
                mx[:, eef0 + 3 : eef1].all(axis=1)
                & my[:, eef0 + 3 : eef1].all(axis=1)
            )
            count = int(pos_valid.sum())
            rot_count = int(rot_valid.sum())
            entry: Dict[str, Any] = {
                "count": count,
                "rot_count": rot_count,
                "fit_pos_sq_sum": 0.0,
                "fit_pos_max_m": 0.0,
                "fit_rot_sum_deg": 0.0,
                "fit_rot_max_deg": 0.0,
                "cal_pos_sq_sum": 0.0,
                "cal_pos_max_m": 0.0,
                "cal_rot_sum_deg": 0.0,
                "cal_rot_max_deg": 0.0,
            }
            if count:
                pos_a = x[pos_valid, eef0 : eef0 + 3].astype(np.float64)
                pos_b = y[pos_valid, eef0 : eef0 + 3].astype(np.float64)
                Rcal = np.asarray(side_cfg["R"], dtype=np.float64)
                tcal = np.asarray(side_cfg["t_m"], dtype=np.float64)

                # Calibration-frame residual: A1 = Rcal @ B1 + tcal.
                expected_cal = (pos_a - tcal) @ Rcal
                dist_cal = np.linalg.norm(pos_b - expected_cal, axis=1)
                entry["cal_pos_sq_sum"] = float(np.sum(dist_cal * dist_cal))
                entry["cal_pos_max_m"] = float(np.max(dist_cal))

                # Per-dataset rigid fit. It absorbs the actual A1 frame offset.
                if count >= 3:
                    Rfit, _tfit, dist_fit = rigid_fit(pos_b, pos_a)
                    entry["fit_pos_sq_sum"] = float(np.sum(dist_fit * dist_fit))
                    entry["fit_pos_max_m"] = float(np.max(dist_fit))

            if rot_count:
                qa = x[rot_valid, eef0 + 3 : eef1].astype(np.float64)
                qb = y[rot_valid, eef0 + 3 : eef1].astype(np.float64)
                na = np.linalg.norm(qa, axis=1)
                nb = np.linalg.norm(qb, axis=1)
                if np.any(np.abs(nb - 1.0) > 1e-4):
                    raise AssertionError(f"non-unit B1 quaternion in {kind}/{side}")
                qa = qa / np.maximum(na[:, None], 1e-12)
                qb = qb / np.maximum(nb[:, None], 1e-12)
                Ra = quat_xyzw_to_matrices(qa)
                Rb = quat_xyzw_to_matrices(qb)
                Rcal = np.asarray(side_cfg["R"], dtype=np.float64)
                Rexp_cal = np.einsum("ij,njk->nik", Rcal, Rb)
                Rerr_cal = np.einsum(
                    "nij,njk->nik", np.transpose(Ra, (0, 2, 1)), Rexp_cal
                )
                ang_cal = rotation_angle_deg(Rerr_cal)
                entry["cal_rot_sum_deg"] = float(np.sum(ang_cal))
                entry["cal_rot_max_deg"] = float(np.max(ang_cal))

                if rot_count >= 3 and count >= 3:
                    Rfit, _tfit, dist_fit = rigid_fit(pos_b, pos_a)
                    Rexp_fit = np.einsum("ij,njk->nik", Rfit, Rb)
                    Rerr_fit = np.einsum(
                        "nij,njk->nik", np.transpose(Ra, (0, 2, 1)), Rexp_fit
                    )
                    ang_fit = rotation_angle_deg(Rerr_fit)
                    entry["fit_rot_sum_deg"] = float(np.sum(ang_fit))
                    entry["fit_rot_max_deg"] = float(np.max(ang_fit))
            result["sides"][f"{kind}_{side}"] = entry
    return result


def empty_quality() -> Dict[str, Any]:
    return {"frames": 0, "sides": {}}


def merge_quality(agg: Dict[str, Any], q: Mapping[str, Any]) -> None:
    for key, s in q.get("sides", {}).items():
        t = agg["sides"].setdefault(
            key,
            {
                "count": 0,
                "rot_count": 0,
                "fit_pos_sq_sum": 0.0,
                "fit_pos_max_m": 0.0,
                "fit_rot_sum_deg": 0.0,
                "fit_rot_max_deg": 0.0,
                "cal_pos_sq_sum": 0.0,
                "cal_pos_max_m": 0.0,
                "cal_rot_sum_deg": 0.0,
                "cal_rot_max_deg": 0.0,
            },
        )
        t["count"] += int(s["count"])
        t["rot_count"] += int(s["rot_count"])
        for prefix in ("fit", "cal"):
            t[f"{prefix}_pos_sq_sum"] += float(s[f"{prefix}_pos_sq_sum"])
            t[f"{prefix}_pos_max_m"] = max(
                t[f"{prefix}_pos_max_m"], float(s[f"{prefix}_pos_max_m"])
            )
            t[f"{prefix}_rot_sum_deg"] += float(s[f"{prefix}_rot_sum_deg"])
            t[f"{prefix}_rot_max_deg"] = max(
                t[f"{prefix}_rot_max_deg"], float(s[f"{prefix}_rot_max_deg"])
            )


def finalize_quality(q: Mapping[str, Any]) -> Dict[str, Any]:
    out = {"frames": q.get("frames", 0), "sides": {}}
    for key, s in q.get("sides", {}).items():
        n = int(s["count"])
        nr = int(s["rot_count"])
        entry: Dict[str, Any] = {"count": n, "rot_count": nr}
        for prefix in ("fit", "cal"):
            entry[f"{prefix}_pos_rms_mm"] = (
                1000.0 * float(np.sqrt(s[f"{prefix}_pos_sq_sum"] / n)) if n else None
            )
            entry[f"{prefix}_pos_max_mm"] = (
                1000.0 * float(s[f"{prefix}_pos_max_m"]) if n else None
            )
            entry[f"{prefix}_rot_mean_deg"] = (
                float(s[f"{prefix}_rot_sum_deg"] / nr) if nr else None
            )
            entry[f"{prefix}_rot_max_deg"] = (
                float(s[f"{prefix}_rot_max_deg"]) if nr else None
            )
        out["sides"][key] = entry
    return out


# ---------------------------------------------------------------------------
# output writing
# ---------------------------------------------------------------------------
def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_output(output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # A1/B1 are the only blocking outputs. Reports/manifest/run.log are
    # derived files and may be replaced after a dry-run.
    for name in ("A1", "B1"):
        path = output / name
        if path.exists() or path.is_symlink():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists; use --overwrite to replace A1/B1"
                )
            _remove_path(path)
    for name in ("reports", "manifest.json", "run.log"):
        _remove_path(output / name)


def log_factory(path: Path):
    f = open(path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        f.write(msg + "\n")
        f.flush()

    return log, f


def _copy_file(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _link_videos(src_dataset: Path, out_dataset: Path) -> None:
    src = src_dataset / "videos"
    if not src.is_dir():
        return
    dst = out_dataset / "videos"
    _remove_path(dst)
    rel = os.path.relpath(src.resolve(), out_dataset.resolve())
    os.symlink(rel, dst)


def _write_episodes_table(
    out_dataset: Path,
    schema: pa.Schema,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    arrays = []
    for name in schema.names:
        values = [row.get(name) for row in rows]
        arrays.append(pa.array(values, type=schema.field(name).type))
    table = pa.Table.from_arrays(arrays, schema=schema)
    path = out_dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def write_dataset_meta(
    src_dataset: Path,
    out_dataset: Path,
    info: Mapping[str, Any],
    episodes_schema: pa.Schema,
    episodes_jsonl: Sequence[Mapping[str, Any]],
    media_manifest: Optional[Mapping[str, Any]],
    results: Sequence[EpisodeResult],
    family: str,
    tier: int,
    condition: str,
    total_chunks: int,
    provenance: Mapping[str, Any],
) -> None:
    meta = out_dataset / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    rows = [r.meta_row for r in results]
    _write_episodes_table(out_dataset, episodes_schema, rows)

    old_json = {int(x.get("episode_index")): dict(x) for x in episodes_jsonl}
    with open(meta / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            d = dict(old_json.get(r.old_episode, {}))
            d["episode_index"] = r.new_episode
            d["ablation"] = {
                "condition": condition,
                "source_episode_index": r.old_episode,
                "family": family,
                "tier": tier,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if media_manifest and isinstance(media_manifest.get("episodes"), dict):
        mm = dict(media_manifest)
        mm["episodes"] = {
            str(r.new_episode): media_manifest["episodes"][str(r.old_episode)]
            for r in results
            if str(r.old_episode) in media_manifest["episodes"]
        }
        write_json(meta / "media_manifest.json", mm)

    new_info = dict(info)
    new_info["total_episodes"] = len(results)
    new_info["total_frames"] = int(sum(r.length for r in results))
    new_info["total_chunks"] = total_chunks
    splits = dict(info.get("splits", {}))
    split_name = next(iter(splits), "train")
    new_info["splits"] = {split_name: f"0:{len(results)}"}
    custom = dict(info.get("custom", {}))
    custom["ablation"] = {
        "condition": condition,
        "input_dataset": str(src_dataset),
        "family": family,
        "tier": tier,
        "selection": "strict_tier1_tier2",
        "eef": (
            "inherited from A (eef_sim_pose, aligned)"
            if condition == "A1"
            else "URDF-root FK from 55D arm joints; R,t not applied"
        ),
        "stats": "inherited from A; not recomputed",
    }
    new_info["custom"] = custom
    write_json(meta / "info.json", new_info)

    for name in ("tasks.parquet", "subtasks.parquet", "stats.json", "input_manifest.json"):
        _copy_file(src_dataset / "meta" / name, meta / name)
    for name in ("source_metadata", "video_timing"):
        _copy_tree(src_dataset / "meta" / name, meta / name)
    _copy_tree(src_dataset / "qc", out_dataset / "qc")

    prov = dict(provenance)
    prov["output_dataset"] = str(out_dataset)
    prov["condition"] = condition
    write_json(meta / "ablation_provenance.json", prov)
    write_json(
        out_dataset / "qc" / "ablation_provenance.json",
        prov,
    )

    _link_videos(src_dataset, out_dataset)


# ---------------------------------------------------------------------------
# dataset processing
# ---------------------------------------------------------------------------
def process_dataset(
    dataset_dir: Path,
    out_a_root: Path,
    out_b_root: Path,
    calibration: Mapping[str, Any],
    urdf_root: Path,
    dry_run: bool,
    log,
) -> Dict[str, Any]:
    name = dataset_dir.name
    record: Dict[str, Any] = {
        "name": name,
        "path": str(dataset_dir),
        "status": "error",
        "reason": None,
        "family": None,
        "tier": None,
        "candidate_episodes": 0,
        "candidate_frames": 0,
        "episodes": 0,
        "frames": 0,
        "failures": [],
        "quality": empty_quality(),
        "episode_map": [],
    }

    try:
        info = read_json(dataset_dir / "meta" / "info.json")
        input_manifest_path = dataset_dir / "meta" / "input_manifest.json"
        input_manifest = read_json(input_manifest_path) if input_manifest_path.is_file() else {}
        episodes_table = load_episodes_table(dataset_dir)
        episodes = [dict(x) for x in episodes_table.to_pylist()]
        episodes.sort(key=lambda x: int(x["episode_index"]))
        episodes_jsonl = load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        media_path = dataset_dir / "meta" / "media_manifest.json"
        media_manifest = read_json(media_path) if media_path.is_file() else None

        record["candidate_episodes"] = len(episodes)
        record["candidate_frames"] = int(sum(int(x["length"]) for x in episodes))

        n_left, n_right = detect_dof(info, input_manifest)
        family, reason, _matched = resolve_family(
            name, info, input_manifest, n_left, n_right
        )
        record["family"] = family
        record["dof"] = [n_left, n_right]
        if reason:
            record["status"] = "excluded"
            record["reason"] = reason
            log(f"[exclude] {name}: {reason} (DoF={n_left}/{n_right})")
            return record
        if family not in calibration:
            record["status"] = "excluded"
            record["reason"] = "family_not_in_calibration"
            log(f"[exclude] {name}: family {family!r} not in calibration")
            return record
        tier = tier_for(family)
        if tier is None:
            record["status"] = "excluded"
            record["reason"] = "family_not_in_strict_tier"
            log(f"[exclude] {name}: family {family!r} has no tier")
            return record
        record["tier"] = tier

        cfg = calibration[family]
        urdf = Urdf(urdf_root / cfg["urdf"])
        validate_config(cfg, urdf, n_left, n_right)

        if dry_run:
            # Quick preflight: one episode, no writes.
            if episodes:
                reader = SourceReader(dataset_dir, info)
                src = reader.read_episode(episodes[0])
                common = reindex_table(src, 0, 0)
                b1, _ = build_b1_table(common, cfg, urdf, n_left, n_right)
                validate_pair(common, b1, cfg)
            record["status"] = "included"
            record["reason"] = "dry_run"
            record["episodes"] = len(episodes)
            record["frames"] = record["candidate_frames"]
            log(
                f"[dry-run] {name}: family={family} tier={tier} "
                f"episodes={len(episodes)} frames={record['candidate_frames']}"
            )
            return record

        out_a = out_a_root / name
        out_b = out_b_root / name
        _remove_path(out_a)
        _remove_path(out_b)
        out_a.mkdir(parents=True, exist_ok=True)
        out_b.mkdir(parents=True, exist_ok=True)

        reader = SourceReader(dataset_dir, info)
        has_offsets = {
            "data/file_from_offset",
            "data/file_to_offset",
        }.issubset(set(episodes_table.schema.names))
        data_path_template = str(info.get("data_path", ""))
        per_episode_template = (
            "{episode_index" in data_path_template
            or "{episode_chunk" in data_path_template
        )
        writer = PairedBufferedWriter(
            out_a,
            out_b,
            info,
            one_file_per_episode=(not has_offsets) or per_episode_template,
        )
        progress = Progress(
            record["candidate_frames"],
            desc=f"Build {name}",
            unit="frames",
        )
        results: List[EpisodeResult] = []
        failures: List[Dict[str, Any]] = []
        quality = empty_quality()
        new_global = 0

        for ep in episodes:
            old_ep = int(ep["episode_index"])
            length = int(ep["length"])
            try:
                src = reader.read_episode(ep)
                new_ep = len(results)
                common = reindex_table(src, new_ep, new_global)
                b1, _ = build_b1_table(common, cfg, urdf, n_left, n_right)
                q = validate_pair(common, b1, cfg)
            except Exception as exc:  # episode-level failure: skip both A1/B1
                failures.append(
                    {"episode": old_ep, "error": f"{type(exc).__name__}: {exc}"}
                )
                progress.update(length, suffix=f"{name} ep {old_ep} FAILED")
                continue

            meta_row = dict(ep)
            meta_row["episode_index"] = new_ep
            meta_row["dataset_from_index"] = new_global
            meta_row["dataset_to_index"] = new_global + length
            meta_row["meta/episodes/chunk_index"] = 0
            meta_row["meta/episodes/file_index"] = 0
            result = EpisodeResult(
                old_episode=old_ep,
                new_episode=new_ep,
                length=length,
                original_row=ep,
                meta_row=meta_row,
                a1_table=common,
                b1_table=b1,
                quality=q,
            )
            writer.add(result)
            results.append(result)
            # Release episode tables; the writer owns them until flush.
            result.a1_table = None
            result.b1_table = None
            merge_quality(quality, q)
            new_global += length
            progress.update(length, suffix=f"{name} ep {old_ep}")

        progress.close()
        writer.close()
        record["failures"] = failures
        record["episodes"] = len(results)
        record["frames"] = int(sum(r.length for r in results))
        record["quality"] = quality
        record["episode_map"] = [
            {
                "old_episode_index": r.old_episode,
                "new_episode_index": r.new_episode,
                "length": r.length,
            }
            for r in results
        ]

        if not results:
            record["status"] = "excluded"
            record["reason"] = "all_episodes_failed"
            _remove_path(out_a)
            _remove_path(out_b)
            log(f"[exclude] {name}: all episodes failed ({len(failures)} failures)")
            return record

        provenance = {
            "input_dataset": str(dataset_dir),
            "output_dataset": str(out_a),
            "family": family,
            "tier": tier,
            "condition": "A1/B1",
            "urdf": cfg["urdf"],
            "old_to_new": record["episode_map"],
            "failures": failures,
            "stats": "inherited from A; EEF distribution is not recomputed",
        }
        write_dataset_meta(
            dataset_dir,
            out_a,
            info,
            episodes_table.schema,
            episodes_jsonl,
            media_manifest,
            results,
            family,
            tier,
            "A1",
            writer.total_chunks,
            provenance,
        )
        write_dataset_meta(
            dataset_dir,
            out_b,
            info,
            episodes_table.schema,
            episodes_jsonl,
            media_manifest,
            results,
            family,
            tier,
            "B1",
            writer.total_chunks,
            provenance,
        )

        record["status"] = "included"
        log(
            f"[done] {name}: family={family} tier={tier} "
            f"episodes={len(results)}/{len(episodes)} frames={record['frames']}"
        )
        return record

    except Exception as exc:
        record["status"] = "error"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        log(f"[error] {name}: {type(exc).__name__}: {exc}")
        return record


# ---------------------------------------------------------------------------
# reports / main
# ---------------------------------------------------------------------------
def write_reports(
    output: Path,
    input_root: Path,
    urdf_root: Path,
    calibration_path: Path,
    records: Sequence[Mapping[str, Any]],
    elapsed: float,
) -> None:
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    eligibility = {
        "input": str(input_root),
        "policy": "strict_tier1_tier2",
        "urdf_root": str(urdf_root),
        "calibration": str(calibration_path),
        "datasets": list(records),
    }
    write_json(reports / "eligibility.json", eligibility)

    manifest = {
        "input": str(input_root),
        "output": str(output),
        "policy": "strict_tier1_tier2",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": [
            {
                "name": r["name"],
                "status": r["status"],
                "family": r["family"],
                "tier": r["tier"],
                "episodes": r["episodes"],
                "frames": r["frames"],
                "candidate_episodes": r["candidate_episodes"],
                "candidate_frames": r["candidate_frames"],
                "failures": r["failures"],
                "episode_map": r["episode_map"],
            }
            for r in records
        ],
    }
    write_json(output / "manifest.json", manifest)

    fk_quality = {
        "datasets": [
            {
                "name": r["name"],
                "family": r["family"],
                "tier": r["tier"],
                "status": r["status"],
                "quality": finalize_quality(r["quality"]),
            }
            for r in records
            if r["status"] == "included"
        ]
    }
    write_json(reports / "fk_quality.json", fk_quality)

    included = [r for r in records if r["status"] == "included"]
    excluded = [r for r in records if r["status"] == "excluded"]
    errored = [r for r in records if r["status"] == "error"]
    lines = [
        "# A1/B1 build summary",
        "",
        f"- Input: `{input_root}`",
        f"- Output: `{output}`",
        f"- Policy: `strict_tier1_tier2`",
        f"- Elapsed: {elapsed:.1f}s",
        "",
        f"Included datasets: **{len(included)}**",
        f"Excluded datasets: **{len(excluded)}**",
        f"Errored datasets: **{len(errored)}**",
        "",
        "## Included",
        "",
        "| dataset | family | tier | episodes | frames |",
        "|---|---|---:|---:|---:|",
    ]
    for r in included:
        lines.append(
            f"| {r['name']} | {r['family']} | {r['tier']} | "
            f"{r['episodes']} | {fmt_int(r['frames'])} |"
        )
    if excluded:
        lines += ["", "## Excluded", "", "| dataset | reason |", "|---|---|"]
        for r in excluded:
            lines.append(f"| {r['name']} | {r['reason']} |")
    if errored:
        lines += ["", "## Errors", "", "| dataset | error |", "|---|---|"]
        for r in errored:
            lines.append(f"| {r['name']} | {r['reason']} |")
    lines += [
        "",
        "## Notes",
        "",
        "- A1 keeps A's aligned EEF (`eef_sim_pose`) unchanged.",
        "- B1 replaces EEF 14:28 with URDF-root FK; `R,t` is NOT applied.",
        "- Videos are symlinked to A; A1/B1 do not duplicate video files.",
        "- `stats.json` is inherited from A and is not recomputed.",
        "- Tier 2 left-arm residuals are documented in `fk_quality.json`.",
        "",
    ]
    (reports / "diff_summary.md").write_text("\n".join(lines), encoding="utf-8")

    urdf_files = []
    for r in included:
        fam = r["family"]
        # The URDF path is recorded in eligibility; re-read from calibration.
        try:
            cfg = read_json(calibration_path)["families"][fam]
            urdf_files.append(cfg["urdf"])
        except Exception:
            pass
    provenance = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "urdf_root": str(urdf_root),
        "urdf_files": sorted(set(urdf_files)),
        "input": str(input_root),
        "policy": "strict_tier1_tier2",
        "note": "R,t is used only for validation; it is not applied to B1.",
    }
    write_json(reports / "provenance.json", provenance)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build paired A1 (A subset) and B1 (URDF-root FK EEF) datasets."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="full processed A root, or one LeRobot v3.0 dataset (default: ./A)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="output root; A1/, B1/, reports/ are created (default: ./ablation)",
    )
    parser.add_argument(
        "--urdf-root",
        default=str(DEFAULT_URDF_ROOT),
        help="URDF root (default: <script_dir>/urdf)",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="calibration.json (default: <urdf-root>/validation/calibration.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="scan only; write reports, no A1/B1 data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing A1/B1/reports")
    args = parser.parse_args(argv)

    input_root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    urdf_root = Path(args.urdf_root).expanduser().resolve()
    calibration_path = (
        Path(args.calibration).expanduser().resolve()
        if args.calibration
        else (urdf_root / "validation" / "calibration.json").resolve()
    )

    if not input_root.is_dir():
        print(f"error: input not found: {input_root}", file=sys.stderr)
        return 2
    if not calibration_path.is_file():
        print(f"error: calibration not found: {calibration_path}", file=sys.stderr)
        return 2
    if not urdf_root.is_dir():
        print(f"error: urdf root not found: {urdf_root}", file=sys.stderr)
        return 2

    try:
        prepare_output(output, args.overwrite)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    log, log_file = log_factory(output / "run.log")
    t0 = time.monotonic()
    try:
        calibration = read_json(calibration_path)["families"]
        datasets = find_datasets(input_root)
        log(
            f"input={input_root}\noutput={output}\n"
            f"urdf_root={urdf_root}\ncalibration={calibration_path}\n"
            f"datasets={len(datasets)} policy=strict_tier1_tier2 dry_run={args.dry_run}"
        )

        records = []
        for dataset_dir in datasets:
            records.append(
                process_dataset(
                    dataset_dir,
                    output / "A1",
                    output / "B1",
                    calibration,
                    urdf_root,
                    args.dry_run,
                    log,
                )
            )

        elapsed = time.monotonic() - t0
        write_reports(
            output,
            input_root,
            urdf_root,
            calibration_path,
            records,
            elapsed,
        )

        included = [r for r in records if r["status"] == "included"]
        total_frames = sum(r["frames"] for r in included)
        log(
            f"\nFinished in {elapsed:.1f}s | included datasets={len(included)} | "
            f"episodes={sum(r['episodes'] for r in included):,} | "
            f"frames={total_frames:,}"
        )
        log(f"output: {output}")
        return 0 if not any(r["status"] == "error" for r in records) else 1
    finally:
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
