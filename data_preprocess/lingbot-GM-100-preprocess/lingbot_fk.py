#!/usr/bin/env python
"""Forward kinematics for the lingbot GM-100 arms, using the Piper-arm URDF.

Same robot rig as the GM100 dataset; this module is carried over unchanged
from gm100_fk.py (only the class name differs), with the same validated
conventions.

Loads `urdf/aloha_tracer2_dabai_dark.urdf` and computes the end-effector pose
of the two working arms:

    fl = left arm   (URDF joint chain fl_base_joint -> fl_joint1..6 -> fl_link6)
    fr = right arm  (URDF joint chain fr_base_joint -> fr_joint1..6 -> fr_link6)

eef = the wrist-flange frame (link6), i.e. the frame the gripper fingers are
mounted on. Anchor frame = URDF root ("footprint"), which is coincident with
"body_Link" where both arms are mounted (body_joint origin is identity) --
this is the robot-base-frame analog used by the 55D spec for robot datasets.

Joint mapping (validated against URDF joint limits; see GM100 project README §3.2):
    observation.state.arm.position[0:6]  -> fl_joint1..6   (left arm)
    observation.state.arm.position[6:12] -> fr_joint1..6   (right arm)
Same zero reference and sign convention as the URDF limits (evidence: J2 is
positive-only and J3 negative-only in both the URDF limits and the data).

Conventions:
  - input joint angles: radians
  - output position: meters (3,); orientation: quaternion xyzw (4,) -- the 55D
    quaternion order convention
  - URDF RPY = Rx(roll) then Ry(pitch) then Rz(yaw), fixed axes
    (scipy equivalent: Rotation.from_euler("xyz", [r, p, y]))
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_URDF = Path(__file__).resolve().parent / "urdf" / "aloha_tracer2_dabai_dark.urdf"


def _rot_rpy(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_matrix()


def _rot_axis_angle(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(N,) joint angles -> (N,3,3) rotations about a fixed axis (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return (np.eye(3)[None]
            + np.sin(theta)[:, None, None] * K[None]
            + (1.0 - np.cos(theta))[:, None, None] * (K @ K)[None])


def _T(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rot_rpy(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


class ArmFK:
    """FK for one arm: fixed base joint (body -> arm mount) + 6 revolute joints."""

    def __init__(self, base_xyz, base_rpy, joints):
        self.T_base = _T(base_xyz, base_rpy)
        self.fixed = [_T(xyz, rpy) for xyz, rpy, _ in joints]
        self.axes = [np.asarray(ax, dtype=np.float64) for _, _, ax in joints]

    def fk(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """angles (N,6) rad -> (pos (N,3) m, quat (N,4) xyzw)."""
        angles = np.atleast_2d(np.asarray(angles, dtype=np.float64))
        n = angles.shape[0]
        T = np.broadcast_to(self.T_base, (n, 4, 4)).copy()
        for i in range(6):
            R = _rot_axis_angle(self.axes[i], angles[:, i])          # (n,3,3)
            M = np.zeros((n, 4, 4))
            M[:, :3, :3] = R
            M[:, 3, 3] = 1.0
            # URDF joint transform: origin (fixed) then rotation about axis
            T = T @ (self.fixed[i] @ M)
        pos = T[:, :3, 3]
        quat = Rotation.from_matrix(T[:, :3, :3]).as_quat()          # xyzw
        return pos, quat


def load_arm(urdf_path: Path, prefix: str) -> ArmFK:
    root = ET.parse(str(urdf_path)).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}

    def origin_of(joint) -> tuple[np.ndarray, np.ndarray]:
        o = joint.find("origin")
        xyz = np.array([float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()])
        rpy = np.array([float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()])
        return xyz, rpy

    base_xyz, base_rpy = origin_of(by_name[f"{prefix}_base_joint"])
    joints = []
    for i in range(1, 7):
        j = by_name[f"{prefix}_joint{i}"]
        assert j.get("type") == "revolute", f"{prefix}_joint{i} is {j.get('type')}"
        xyz, rpy = origin_of(j)
        ax = j.find("axis")
        axis = np.array([float(v) for v in ((ax.get("xyz") if ax is not None else None) or "0 0 1").split()])
        joints.append((xyz, rpy, axis))
    return ArmFK(base_xyz, base_rpy, joints)


class LingbotFK:
    """Both arms of the lingbot GM-100 rig (Piper arms; left=fl, right=fr)."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF):
        self.urdf_path = Path(urdf_path)
        self.left = load_arm(self.urdf_path, "fl")
        self.right = load_arm(self.urdf_path, "fr")

    def fk_left(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.left.fk(angles)

    def fk_right(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.right.fk(angles)


if __name__ == "__main__":
    # --- self-test: conventions and sanity ---
    # 1) RPY convention: rpy=[0,0,pi/2] must map x-axis onto y-axis
    R = _rot_rpy(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12), "RPY convention broken"
    # 2) Rodrigues about z
    Rz = _rot_axis_angle(np.array([0.0, 0.0, 1.0]), np.array([np.pi / 2]))[0]
    assert np.allclose(Rz @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12), "axis-angle broken"

    fk = LingbotFK()
    zero = np.zeros((1, 6))
    for name, fn in (("left(fl)", fk.fk_left), ("right(fr)", fk.fk_right)):
        pos, quat = fn(zero)
        qn = float(np.linalg.norm(quat[0]))
        print(f"{name} @ zero pose: pos={np.round(pos[0], 4).tolist()} m  "
              f"quat_xyzw={np.round(quat[0], 4).tolist()}  |q|={qn:.6f}")
        assert abs(qn - 1.0) < 1e-9
        assert np.linalg.norm(pos[0]) < 1.5
    print("self-test OK")
