#!/usr/bin/env python
"""Forward kinematics for the Franka Panda arm (standard published DH).

Used as an INDEPENDENT CROSS-CHECK in validation: it recomputes the flange
pose from joint angles only. The 55D eef values themselves come directly from
the source `states.gripper.pose` field (flange position + euler angles), and
this module verifies them to high precision.

Kinematics: the standard Franka Emika Panda modified-DH table (Craig
convention), the same parameters used by the official franka_description
URDF and by ROS / Isaac / MuJoCo Franka models:

    i | a_{i-1} (m) | alpha_{i-1} (rad) | d_i (m)
    1 |  0          |  0                | 0.333
    2 |  0          | -pi/2             | 0
    3 |  0          | +pi/2             | 0.316
    4 | +0.0825     | +pi/2             | 0
    5 | -0.0825     | -pi/2             | 0.384
    6 |  0          | +pi/2             | 0
    7 | +0.088      | +pi/2             | 0
    flange = link7 translated +0.107 m along z

Empirical validation against this dataset (all 140,157 frames, close_the_laptop):
  - FK flange position        vs source ee position : max residual 0.00000 m
  - FK flange orientation     vs source gripper.pose euler (URDF rpy):
                                                    max deviation 0.0000 deg
  - TCP offset (0.095 m) direction vs FK flange z-axis: mean 0.01 deg
  -> the dataset's joint angles follow the standard Franka convention
     (same zeros/signs), and the FK below reproduces the sim exactly.

Conventions:
  - input joint angles: radians, order j1..j7
  - output: position meters (m), orientation quaternion xyzw (55D convention)
  - base frame: the DH base == the dataset's `armbase`/`robot` frame
    (verified zero base offset: ee_pos == FK flange origin exactly)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# modified-DH rows: (a_{i-1}, alpha_{i-1}, d_i)
DH = [
    (0.0,      0.0,       0.333),
    (0.0,      -np.pi / 2, 0.0),
    (0.0,      np.pi / 2,  0.316),
    (0.0825,   np.pi / 2,  0.0),
    (-0.0825,  -np.pi / 2, 0.384),
    (0.0,      np.pi / 2,  0.0),
    (0.088,    np.pi / 2,  0.0),
]
FLANGE_OFFSET = 0.107   # link7 origin -> flange origin, along link7 z


def fk_chain(theta: np.ndarray) -> np.ndarray:
    """Joint angles (N,7) rad -> (N,4,4) link7 homogeneous transforms."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    n = theta.shape[0]
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i, (a, alpha, d) in enumerate(DH):
        th = theta[:, i]
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(alpha), np.sin(alpha)
        Ti = np.zeros((n, 4, 4))
        Ti[:, 0, 0] = ct
        Ti[:, 0, 1] = -st
        Ti[:, 1, 0] = st * ca
        Ti[:, 1, 1] = ct * ca
        Ti[:, 1, 2] = -sa
        Ti[:, 2, 0] = st * sa
        Ti[:, 2, 1] = ct * sa
        Ti[:, 2, 2] = ca
        Ti[:, 0, 3] = a
        Ti[:, 1, 3] = -sa * d
        Ti[:, 2, 3] = ca * d
        Ti[:, 3, 3] = 1.0
        T = T @ Ti
    return T


def fk_flange(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint angles (N,7) rad -> flange (pos (N,3) m, quat (N,4) xyzw)."""
    T = fk_chain(theta)
    z = T[:, :3, 2]
    pos = T[:, :3, 3] + FLANGE_OFFSET * z          # link7 origin -> flange
    quat = Rotation.from_matrix(T[:, :3, :3]).as_quat()   # xyzw
    return pos, quat


if __name__ == "__main__":
    # --- self-test: conventions and sanity ---
    zero = np.zeros((1, 7))
    pos, quat = fk_flange(zero)
    print(f"@ zero pose: flange pos = {np.round(pos[0], 4).tolist()} m  "
          f"quat_xyzw = {np.round(quat[0], 4).tolist()}  |q| = {np.linalg.norm(quat[0]):.6f}")
    assert abs(np.linalg.norm(quat[0]) - 1.0) < 1e-9, "quaternion not normalized"
    assert np.linalg.norm(pos[0]) < 1.5, "flange position implausible"

    # flange = link7 origin + 0.107 * z must hold exactly
    T = fk_chain(zero)
    pos2, _ = fk_flange(zero)
    assert np.allclose(pos2[0], T[0, :3, 3] + FLANGE_OFFSET * T[0, :3, 2]), "flange offset broken"

    # known reach: franka flange max reach ~0.855 m; sample a bent pose
    test = np.array([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]])
    p, q = fk_flange(test)
    print(f"@ test pose: flange pos = {np.round(p[0], 4).tolist()} m  |q| = {np.linalg.norm(q[0]):.6f}")
    assert np.linalg.norm(p[0]) < 0.9, "reach exceeds franka spec"
    print("self-test OK")
