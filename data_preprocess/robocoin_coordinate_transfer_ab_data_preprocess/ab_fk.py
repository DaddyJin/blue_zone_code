"""Minimal URDF parser and vectorized forward kinematics.

Only the subset needed by this package is implemented:
  * links / joints / parent-child tree
  * revolute, continuous, prismatic, fixed joints
  * joint origin (xyz, rpy), axis, limits are ignored for FK
  * no mesh loading, no inertia, no mimic joints

The FK convention matches the calibration probe used to create
``urdf/validation/calibration.json``:

    T_parent<-child(q) = T_origin @ T_motion(q)

For a revolute joint, ``T_motion`` is a rotation around ``axis`` by ``q``;
for a prismatic joint it is a translation along ``axis`` by ``q``.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw -> 3x3 rotation matrix.

    This is the same R = Rz(yaw) @ Ry(pitch) @ Rx(roll) convention used by
    the calibration tooling.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_angle_matrices(axis: Sequence[float], q: np.ndarray) -> np.ndarray:
    """Return (N, 3, 3) rotation matrices for rotation about ``axis`` by q."""
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        a = a / norm
    K = np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        dtype=np.float64,
    )
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    s = np.sin(q)[:, None, None]
    c = (1.0 - np.cos(q))[:, None, None]
    return np.eye(3, dtype=np.float64)[None, :, :] + s * K[None, :, :] + c * (K @ K)[None, :, :]


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray  # 3


def _parse_origin(node: Optional[ET.Element]) -> np.ndarray:
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if node is not None:
        if node.get("xyz"):
            xyz = [float(x) for x in node.get("xyz", "0 0 0").split()]
        if node.get("rpy"):
            rpy = [float(x) for x in node.get("rpy", "0 0 0").split()]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def _parse_joint(joint: ET.Element) -> JointSpec:
    parent_node = joint.find("parent")
    child_node = joint.find("child")
    if parent_node is None or child_node is None:
        raise ValueError(f"joint {joint.get('name')!r} has no parent/child")
    axis_node = joint.find("axis")
    axis = [1.0, 0.0, 0.0]
    if axis_node is not None and axis_node.get("xyz"):
        axis = [float(x) for x in axis_node.get("xyz", "1 0 0").split()]
    return JointSpec(
        name=joint.get("name", ""),
        kind=joint.get("type", "fixed"),
        parent=parent_node.get("link", ""),
        child=child_node.get("link", ""),
        origin=_parse_origin(joint.find("origin")),
        axis=np.asarray(axis, dtype=np.float64),
    )


class Urdf:
    """Parsed URDF tree with cached FK chains."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        root = ET.parse(self.path).getroot()
        joints = [_parse_joint(j) for j in root.findall("joint")]
        self.joints: Dict[str, JointSpec] = {j.name: j for j in joints}
        self.by_child: Dict[str, JointSpec] = {j.child: j for j in joints}
        children = {j.child for j in joints}
        parents = {j.parent for j in joints}
        roots = sorted(parents - children)
        if len(roots) != 1:
            # URDFs normally have exactly one root. If not, leave it unset and
            # let the chain walk find a link without a parent joint.
            self.root_link = roots[0] if roots else ""
        else:
            self.root_link = roots[0]
        self._chain_cache: Dict[str, Tuple[Tuple[JointSpec, ...], str]] = {}

    def chain(self, target_link: str) -> Tuple[Tuple[JointSpec, ...], str]:
        """Return (joints from root to target, root link)."""
        if target_link in self._chain_cache:
            return self._chain_cache[target_link]
        path = []
        link = target_link
        seen = set()
        while link in self.by_child:
            if link in seen:
                raise ValueError(f"cycle in URDF at link {link!r}")
            seen.add(link)
            joint = self.by_child[link]
            path.append(joint)
            link = joint.parent
        if not path and target_link != self.root_link:
            raise KeyError(f"link {target_link!r} not found in {self.path}")
        result = (tuple(reversed(path)), link)
        self._chain_cache[target_link] = result
        return result

    def link_names(self) -> set[str]:
        names = {j.parent for j in self.joints.values()} | {j.child for j in self.joints.values()}
        return names

    def fk_batch(
        self,
        qmap: Mapping[str, np.ndarray | float],
        target_link: str,
        n: Optional[int] = None,
    ) -> Tuple[np.ndarray, str]:
        """Vectorized FK.

        Parameters
        ----------
        qmap
            joint name -> scalar or (N,) array. Missing joints are treated as 0.
        target_link
            Link whose pose is returned.
        n
            Number of frames. If omitted, inferred from the first array in qmap.

        Returns
        -------
        T : (N, 4, 4) transforms of target_link relative to the URDF root.
        root : root link name.
        """
        chain, root = self.chain(target_link)
        if n is None:
            for value in qmap.values():
                if np.ndim(value) > 0:
                    n = int(np.asarray(value).shape[0])
                    break
        if n is None:
            n = 1
        T = np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()

        for joint in chain:
            value = qmap.get(joint.name, 0.0)
            if np.ndim(value) == 0:
                q = np.full(n, float(value), dtype=np.float64)
            else:
                q = np.asarray(value, dtype=np.float64).reshape(-1)
                if q.shape[0] != n:
                    raise ValueError(
                        f"joint {joint.name!r}: expected {n} values, got {q.shape[0]}"
                    )

            M = np.broadcast_to(joint.origin, (n, 4, 4)).copy()
            J = np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()
            if joint.kind in ("revolute", "continuous"):
                J[:, :3, :3] = axis_angle_matrices(joint.axis, q)
            elif joint.kind == "prismatic":
                axis = np.asarray(joint.axis, dtype=np.float64)
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    axis = axis / norm
                J[:, :3, 3] = q[:, None] * axis[None, :]
            elif joint.kind == "fixed":
                pass
            else:
                # floating/planar are not used by the included robots; treat as fixed.
                pass
            T = T @ M @ J
        return T, root


def rotation_matrices_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 3) rotation matrices to (N, 4) xyzw unit quaternions."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R must be (N,3,3), got {R.shape}")
    n = R.shape[0]
    q = np.zeros((n, 4), dtype=np.float64)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    trace = m00 + m11 + m22
    done = np.zeros(n, dtype=bool)

    sel = trace > 0
    if np.any(sel):
        S = np.sqrt(np.maximum(trace[sel] + 1.0, 0.0)) * 2.0
        q[sel, 0] = (m21[sel] - m12[sel]) / S
        q[sel, 1] = (m02[sel] - m20[sel]) / S
        q[sel, 2] = (m10[sel] - m01[sel]) / S
        q[sel, 3] = 0.25 * S
        done[sel] = True

    sel = (~done) & (m00 >= m11) & (m00 >= m22)
    if np.any(sel):
        S = np.sqrt(np.maximum(1.0 + m00[sel] - m11[sel] - m22[sel], 0.0)) * 2.0
        q[sel, 0] = 0.25 * S
        q[sel, 1] = (m01[sel] + m10[sel]) / S
        q[sel, 2] = (m02[sel] + m20[sel]) / S
        q[sel, 3] = (m21[sel] - m12[sel]) / S
        done[sel] = True

    sel = (~done) & (m11 >= m22)
    if np.any(sel):
        S = np.sqrt(np.maximum(1.0 + m11[sel] - m00[sel] - m22[sel], 0.0)) * 2.0
        q[sel, 0] = (m01[sel] + m10[sel]) / S
        q[sel, 1] = 0.25 * S
        q[sel, 2] = (m12[sel] + m21[sel]) / S
        q[sel, 3] = (m02[sel] - m20[sel]) / S
        done[sel] = True

    sel = ~done
    if np.any(sel):
        S = np.sqrt(np.maximum(1.0 + m22[sel] - m00[sel] - m11[sel], 0.0)) * 2.0
        q[sel, 0] = (m02[sel] + m20[sel]) / S
        q[sel, 1] = (m12[sel] + m21[sel]) / S
        q[sel, 2] = 0.25 * S
        q[sel, 3] = (m10[sel] - m01[sel]) / S

    norm = np.linalg.norm(q, axis=1)
    bad = (norm < 1e-12) | ~np.isfinite(norm)
    q[~bad] = q[~bad] / norm[~bad, None]
    q[bad] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q


def quat_sign_continuity(q: np.ndarray) -> np.ndarray:
    """Flip q[i] to keep dot(q[i], q[i-1]) >= 0.

    The quaternion sign is physically irrelevant; this only avoids visual and
    numerical discontinuities within an episode.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    if len(q) <= 1:
        return q
    dots = np.sum(q[1:] * q[:-1], axis=1)
    flips = dots < 0.0
    signs = np.ones(len(q), dtype=np.float64)
    signs[1:] = np.cumprod(np.where(flips, -1.0, 1.0))
    return q * signs[:, None]


def pack_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split (N,4,4) transforms into positions and xyzw quaternions."""
    T = np.asarray(T, dtype=np.float64)
    pos = T[:, :3, 3]
    quat = quat_sign_continuity(rotation_matrices_to_quat_xyzw(T[:, :3, :3]))
    return pos, quat


def rigid_fit(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Q ~= R @ P + t with Umeyama/SVD.

    Returns (R, t, per-point residuals). Both P and Q are (N,3) positions.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"expected matching (N,3), got {P.shape} and {Q.shape}")
    if len(P) < 3:
        raise ValueError("rigid_fit needs at least 3 points")
    muP = P.mean(axis=0)
    muQ = Q.mean(axis=0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = muQ - R @ muP
    res = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
    return R, t, res
