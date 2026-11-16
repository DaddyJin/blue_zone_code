# calibration.json 说明

`urdf/validation/calibration.json` 是本包的核心配置。它不是 URDF，也不是数据，
而是**每个机器人家族的“用哪个模型、怎么算、算得准不准”**。

它由 `probe_eef_sim.py` 生成：

1. 取 RoboCOIN 的 `observation.state` 关节角；
2. 用候选 URDF 做 FK；
3. 和 `eef_sim_pose_state` 的位置做 Umeyama 刚体拟合；
4. 输出 `R,t`、位置残差、最佳 Euler 表示和姿态偏差。

当前每个家族只用了一个 episode 做验证；正式使用前建议扩到多集。

---

## 1. 顶层结构

```json
{
  "generated_at": "2026-11-13",
  "method": "FK from candidate URDF vs dataset eef_sim_pose_state; rigid fit (Umeyama)",
  "families": {
    "Unitree_G1": { ... }
  }
}
```

---

## 2. 每个 family 的字段

| 字段 | 含义 |
|---|---|
| `urdf` | 推荐 URDF，相对 `urdf/` 根目录 |
| `eef_link.left/right` | 左右 EEF link 名称 |
| `arm_joints.left/right` | URDF 关节名，按数据关节顺序排列 |
| `state_arm_index.left/right` | **原始 `observation.state`** 中臂关节的起始列 |
| `extra_joint_mapping` | 额外的腰/头关节名 -> 原始 state 索引 |
| `note` | 人读的 caveat |
| `sides.L/R` | 左右侧的拟合/残差结果 |

### `sides.L/R`

| 字段 | 含义 |
|---|---|
| `position_rms_m` | FK 与 `eef_sim` 的位置 RMS |
| `position_max_m` | 最大位置残差 |
| `R` | URDF root -> 统一坐标系的拟合旋转 |
| `t_m` | URDF root -> 统一坐标系的拟合平移 |
| `best_rotation_representation` | 最吻合的 Euler 约定 |
| `rotation_mean_dev_deg` / `rotation_max_dev_deg` | 姿态偏差 |
| `rotation_note` | 旋转 caveat |

---

## 3. 例子：Unitree G1

```json
"Unitree_G1": {
  "urdf": "models/Unitree_G1/robocoin_g1_body29_hand14/g1_body29_hand14.urdf",
  "eef_link": {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link"
  },
  "arm_joints": {
    "left": [
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint"
    ],
    "right": [...]
  },
  "state_arm_index": {
    "left": 0,
    "right": 7
  },
  "sides": {
    "L": {
      "position_rms_m": 0.0,
      "position_max_m": 0.0,
      "R": [[1,0,0],[0,1,0],[0,0,1]],
      "t_m": [0,0,0.793],
      "best_rotation_representation": "xyz_ext",
      "rotation_mean_dev_deg": 0.0,
      "rotation_max_dev_deg": 0.0
    }
  }
}
```

---

## 4. 重要：`state_arm_index` 不是 55D 索引

`state_arm_index` 是**原始 `observation.state`** 的列偏移，不是 `state_unified` 的索引。

| 家族 | left | right |
|---|---:|---:|
| Unitree | 0 | 7 |
| Cobot | 0 | 13 |
| Realman | 14 | 0 |
| Airbot | 0 | 6 |
| AI2 | 0 | 13 |

B1 从 `state_unified/action_unified` 取关节时，使用固定 55D 布局：

```text
left  = 0 : len(arm_joints.left)
right = 7 : 7 + len(arm_joints.right)
```

如果直接把 Cobot 的 `right=13` 用在 55D 上，会取错列。

---

## 5. `R,t` 是什么

```text
P_unified = R @ P_urdf_root + t
```

- `R,t` 是拟合结果，不是 URDF 自带；
- 它把 URDF root 下的 EEF 位置映射到 RoboCOIN 的统一坐标系；
- B1 **不应用** `R,t`；只在验证时使用；
- 如果 A 的 `eef_sim` frame 和标定样本不同，`cal_*` 残差会很大，但 `fit_*` 可能仍然很小。

典型值：

| 家族 | R | t (m) |
|---|---|---|
| Kuavo | I | (0,0,0) |
| Unitree | I | (0,0,0.793) |
| Cobot 左 | ≈I | (0.2366, +0.2970, +0.7755) |
| Cobot 右 | ≈I | (0.2381, -0.2998, +0.7765) |

---

## 6. Tier 与排除

| Tier | 家族 | 状态 |
|---|---|---|
| 1 | Kuavo Pro/LB、Cobot Magic/Split ALOHA、Unitree G1 | 位置精确 |
| 2 | Airbot MMK2、Galaxea R1 Lite 6DoF、Realman RMC-AIDA-L、AgiBot G1 | 可用，左臂有 caveat |
| Excluded | Galbot G1、Tianqing A2、Galaxea R1 Lite 7DoF、AI2 AlphaBot 2 | 无模型/模型不匹配 |

严格模式 = Tier 1 + Tier 2。

---

## 7. 额外关节

AgiBot 的 URDF 链包含 `body_joint1/2`，但 data-cleaning 的 55D 腰槽 42:46 全是 0。
因此 B1 从同一行的原始列读取：

```text
state:  source.observation.state[:, 28:30]
action: source.action[:, 28:30]
```

索引来自 `extra_joint_mapping`。AI2 也有额外映射，但 strict 模式排除 AI2。

---

## 8. 已知 caveat

- Cobot 的位置精确（0.35 / 0.19 mm），但 A 的 `eef_sim` 旋转和该 URDF 没有完全复现；
  B1 的旋转是 FK 原生结果。做位置消融没问题，做姿态消融要单独看。
- Tier 2 的左臂残差较大，尤其 AgiBot 左臂约 104 mm；
  这些子数据集仍会生成 B1，但应在 `fk_quality.json` 中按 family/side 分开看。
- 当前 calibration 每个家族只用一个 episode；多集验证会更可靠。
- 如果 A1 的 EEF 四元数 mask 为 0（旧数据只填位置），B1 仍会填四元数，
  但旋转对比的 `rot_count=0`。
