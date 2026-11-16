# 原理说明

## 1. 三个数据集

### A：已有全量数据

由 `data-cleaning` 处理后的 LeRobot v3.0 数据集集合。每个子数据集在原始数据基础上增加了：

```text
state_unified   float32[55]
action_unified  float32[55]
mask_state      float32[55]
mask_action     float32[55]
```

A 的 EEF 14:28 来自 RoboCOIN 的 `eef_sim_pose_state/action`，已经是 RoboCOIN
统一坐标系（x-forward / y-left / z-up，原点在每台机器人自己的 base / 双脚中心）。

### A1：A 的配对子集

- 只保留“能生成可信 B1”的机器人；
- 与 B1 使用完全相同的 episode、行、帧、视频；
- EEF 14:28 **原样保留 A 的值**，不重算；
- 用于和 B1 做逐帧对比。

### B1：去掉对齐的版本

- 与 A1 完全相同的机器人、episode、行、视频；
- 只替换 EEF 14:28 和对应 mask；
- EEF 由**关节角 + URDF 正运动学**重新计算；
- 结果在**每台机器人自己的 URDF root frame**下；
- **不乘 `R,t`**，所以没有对齐到 A 的 base/脚底中心。

---

## 2. 55D 布局

| 下标 | 内容 |
|---|---|
| `0:7` | 左臂关节角（6-DoF 时第 7 维为空，mask=0） |
| `7:14` | 右臂关节角（同上） |
| `14:17` | 左 EEF 位置 xyz |
| `17:21` | 左 EEF 四元数 xyzw |
| `21:24` | 右 EEF 位置 xyz |
| `24:28` | 右 EEF 四元数 xyzw |
| `28:30` | 左右夹爪 |
| `30:42` | 左右手手指 |
| `42:55` | 腰/头/底盘/预留；data-cleaning 当前全部为 0、mask=0 |

B1 只改 `14:28`；A1/B1 的 `0:14`、`28:55` 必须逐元素相同。

---

## 3. joint、link、EEF link

- **link**：刚体，也是坐标系，是 URDF 树的节点；
- **joint**：连接 parent link 和 child link 的运动副，是树的边；
- **EEF link**：FK 最后读取位姿的那个 link。

URDF joint 提供：

```text
origin xyz/rpy   q=0 时 parent -> joint frame 的固定变换
axis xyz         旋转/平移轴
type             revolute / continuous / prismatic / fixed
```

FK 连乘：

```text
T_parent<-child(q) = T_origin @ T_motion(q)
T_root<-eef        = T_root<-j1(q1) @ ... @ T_jn(qn)
```

- revolute：`T_motion = Rot(axis, q)`；
- prismatic：`T_motion = Trans(axis * q)`；
- fixed：`T_motion = I`。

`R,t` **不是** URDF 自带的，而是用 `eef_sim_pose` 和 FK 拟合出的固定变换：

```text
P_unified = R @ P_urdf_root + t
```

B1 生成时不使用 `R,t`。它只用于验证 A1/B1 的关系。

---

## 4. B1 的逐帧算法

对每个 episode 的每一帧：

```text
左臂 q_L = state_unified/action_unified[:, 0 : len(arm_joints.left)]
右臂 q_R = state_unified/action_unified[:, 7 : 7 + len(arm_joints.right)]
```

6-DoF 取前 6 个；第 7 个槽位保持 0/mask=0。7-DoF 取满 7 个。

然后：

```text
T_L = FK(urdf, q_L, arm_joints.left, eef_link.left)
T_R = FK(urdf, q_R, arm_joints.right, eef_link.right)
```

把 `T` 拆成：

```text
pos  = T[:3, 3]
quat = R_to_quat_xyzw(T[:3, :3])   # 归一化 + 相邻帧符号连续
```

写入：

```text
14:17 = pos_L
17:21 = quat_L
21:24 = pos_R
24:28 = quat_R
```

mask 14:28 按“关节有效 + FK 成功”置 1/0。

### action 的语义

- `state` 的关节角来自 `state_unified[:, 0:14]`；
- `action` 的关节角来自 `action_unified[:, 0:14]`；
- **不重新跑 data-cleaning 的 episode policy**；A 的 action 已经是对齐后的最终表示，B1 只替换它的 EEF。

如果 A 的 action 在整集复制策略中被移动到下一帧，B1 也是对该 action 关节角做 FK，因此和 A1 的 action 时间语义一致。

### AgiBot 的额外关节

AgiBot 的 URDF 链包含 `body_joint1/2`，但 data-cleaning 的 55D 腰槽 42:46 全部为 0。
因此 B1 对 AgiBot 的额外关节从同一行的原始列读取：

```text
state:  source.observation.state[:, 28:30]   # body_joint1/2
action: source.action[:, 28:30]
```

索引来自 `calibration.json` 的 `extra_joint_mapping`。
其他家族没有额外关节映射。

---

## 5. calibration.json 是什么

`urdf/validation/calibration.json` 是家族级配置 + 验证结果，不是 RoboCOIN 官方文件。

每个 family 包含：

| 字段 | 含义 |
|---|---|
| `urdf` | 推荐 URDF 相对路径 |
| `eef_link.left/right` | 左右 EEF link |
| `arm_joints.left/right` | URDF 关节名，按数据关节顺序排列 |
| `state_arm_index.left/right` | **原始 `observation.state`** 的臂关节起始列 |
| `extra_joint_mapping` | 额外腰/头关节（AgiBot/AI2） |
| `sides.L/R` | 左右侧的拟合结果与残差 |

### 重要：`state_arm_index` 不是 55D 索引

`state_arm_index` 是原始 `observation.state` 的列偏移。例如：

| 家族 | left | right |
|---|---:|---:|
| Unitree | 0 | 7 |
| Cobot | 0 | 13 |
| Realman | 14 | 0 |
| Airbot | 0 | 6 |

B1 默认从 `state_unified` 取关节，因此使用固定布局：

```text
left  0 : 0+n
right 7 : 7+n
```

**不能**把 `state_arm_index` 直接用在这个 55D 切片上。

### `R,t` 的含义

`sides.L/R` 中的 `R,t_m` 是把 URDF root 下的点映射到 A 的统一坐标系的拟合变换：

```text
P_unified = R @ P_urdf_root + t
```

- Kuavo：`R=I, t=0`；
- Unitree：`R=I, t=(0,0,0.793)`；
- Cobot：`t=(0.237, ±0.298, 0.776)`；
- 其他家族见 calibration.json。

B1 **不应用**它；它只用于：

```text
P_B1_expected = R^T (P_A1 - t)
R_B1_expected = R^T R_A1
```

---

## 6. 资格与 Tier

Tier 是我们用 `eef_sim_pose` 做 ground truth、用公开 URDF 做 FK 后按残差分出来的，
不是 RoboCOIN 官方标签。

### Tier 1：精确级

| 家族 | 位置残差 | 说明 |
|---|---:|---|
| Kuavo Pro / LB | 0 | `R=I, t=0` |
| Cobot Magic / Split ALOHA | 0.35 / 0.19 mm | 位置精确；旋转未完全复现 |
| Unitree G1 | 0 | `R=I, t=(0,0,0.793)` |

### Tier 2：可用级

| 家族 | 右臂 | 左臂 |
|---|---:|---:|
| Airbot MMK2 | 2.7 mm | 5.9 mm |
| Galaxea R1 Lite 6DoF | 0.066 mm | 31.8 mm |
| Realman RMC-AIDA-L | 6.3 mm | 10.3 mm |
| AgiBot G1 | 0.034 mm | 104 mm |

Tier 2 **能生成 B1**，但左臂的零位/装配/标定误差较大，不能当作精确真值。

### Excluded

| 家族 | 原因 |
|---|---|
| Galbot G1 | 没有公开 URDF |
| Tianqing A2 | 没有公开 URDF |
| Galaxea R1 Lite 7DoF | 只有 6-DoF URDF |
| AI2 AlphaBot 2 | 公开 URDF 与实际硬件不匹配 |

严格模式 = Tier 1 + Tier 2；预计约 67.7% 帧。

---

## 7. A1/B1 校验指标

脚本对每个子数据集、每个 kind（state/action）、每个 side 记录两组残差：

### `fit_*`

用当前子数据集的 A1/B1 位置自己拟合一个 `R,t`，然后计算残差。
它能吸收“这个子数据集的 A frame 和 calibration 样本不同”的偏移，主要反映 URDF/模型误差。

### `cal_*`

直接使用 `calibration.json` 的 `R,t` 计算残差。
如果 A 的 `eef_sim` frame 和标定样本不同，这个值会很大（例如 Unitree 某些子数据集差 0.793 m），
这时应看 `fit_*` 是否仍然小。

### 旋转

只有 A1 的四元数 mask 有效时才统计旋转残差。
如果 A1 的 EEF 四元数 mask 为 0（旧数据只填位置），`rot_count=0`，B1 仍然会填四元数。

### action 残差可能大于 state

个别来源的 action 是主臂/命令目标，和 state 的从臂实测值不完全同源；
A 的 `eef_sim_pose_action` 和 action 关节角在某些家族也可能存在微小不一致。
B1 按定义对 **action 关节角**做 FK，因此 `action_*` 的 `fit_*` 残差可能大于
`state_*`。这属于 A 的 action 语义差异，不是 B1 的生成错误；报告会分 kind 列出。

---

## 8. 输出不变量

```text
A1/B1 episode 集合相同
帧数相同
视频 symlink 目标相同
state_unified/action_unified 除 14:28 外完全相同
mask_state/mask_action 除 14:28 外完全相同
reliability 除 14:28 外完全相同（若存在）
timestamp/task_index/frame_index/episode_index 完全相同
只有 EEF 14:28 的值和 mask 可以不同
```

---

## 9. 为什么不做 B2

B2 是“尽量用原始数据集提供的 EEF”。它的 frame、参考点、Euler 约定、时间语义可能每个厂商/仓库都不同，
会引入“对齐”之外的混淆因素。

本脚本只做 B1：

```text
B1 = 关节角 + URDF FK，URDF root frame
```

这样 A1/B1 的差异才能唯一归因于“是否统一到 base/脚底中心”。
