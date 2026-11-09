> **历史流程（v1）**：本文保留已有 EgoSuite 的处理约定；固定尺寸、30fps、时长及特定目录要求不再是通用约束。新数据集请从 [Unified55 v2 主规范](../README.md) 和 [通用工具](../tools/README.md) 开始。

# EgoStandard → 55D Canonical LeRobot v3 格式说明

本文档说明 `egostandard_55d/` 数据集的组织方式、使用方法，以及**透传列**
（原始 mcap 中保留但未处理的数据）的取用方式。目标读者：下游训练、以及要做
手部 retarget 回填的同事。

转换链路：mcap → `hf_tools/egostandard_staging.py`（逐 task staging）
→ `unified_pipeline_v3/batch_robot.py`（canonicalize55 + L1 QC + 官方布局）
→ 视频 500MB 大文件合并（`hf_tools/egostandard_remerge_videos.py`）。

---

## 1. 目录组织（官方 LeRobot v3.0 布局，per-task 一个数据集）

```
egostandard_55d/<子集>__<任务名>/          # 例: EgoStandard-body__File_Archiving
├── data/chunk-000/file-*.parquet          # 逐帧数据，~100MB 轮转
├── videos/<相机>/chunk-000/file-*.mp4     # h264 960x728@30fps，~500MB 轮转
├── meta/
│   ├── info.json                          # codebase_version=v3.0, fps=30, features, custom
│   ├── tasks.parquet                      # 任务文本词表（pandas index=task 文本）
│   ├── subtasks.parquet                   # 子任务词表（pandas index=subtask 文本）
│   ├── episodes.jsonl                     # sidecar: action_config + source_fps 等
│   └── episodes/chunk-000/file-*.parquet  # 每集索引：length、data/videos 定位列、时间戳
└── qc_report.json                         # L1 质检报告
```

`<子集>` 为 `EgoStandard`（无身体姿态）或 `EgoStandard-body`（含 `/pose/body`）。
每个 task 目录就是一个独立可加载的 LeRobot 数据集。

**帧定位**：`meta/episodes` 每行给出 `data/chunk_index|file_index` 与每个相机的
`videos/.../chunk_index|file_index|from_timestamp|to_timestamp`（合并后的大文件 +
片内时间戳，秒）。`from/to_timestamp` 是连续累加的容器时间轴。

### 加载示例

```python
import os
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
from lerobot.datasets.lerobot_dataset import LeRobotDataset   # 见第 6 节的导入顺序坑

ds = LeRobotDataset("local/xxx", root="egostandard_55d/EgoStandard-body__File_Archiving")
row = ds.hf_dataset[0]                    # 不走视频解码，拿全部列
tasks_df = ds.meta.tasks                  # index=任务文本
task_txt = tasks_df.iloc[row["task_index"].item()].name
```

视频解码建议直接用 PyAV 按 episodes 索引列定位（参考
`hf_tools/egostandard_55d_official_load.py` 的 `decode_at()`）——本环境
torchcodec/torchvision 视频后端均不可用。

## 2. 55D 统一动作/状态槽位（state_unified / action_unified）

固定 55 维 float32，槽位布局（`unified_pipeline_v3/.../schema/canonical.py`）：

| 维度 | 槽位 | EgoStandard 填的什么 |
|---|---|---|
| 0–13 | left/right_arm_joint | 空（mask=0，源数据无关节角） |
| 14–20 | left_eef | 左腕 pos(3)+quat(4, **xyzw**)，逐帧躯干偏航系 |
| 21–27 | right_eef | 右腕同上 |
| 28 | left_gripper | 左手捏合度 [0,1]（拇指尖-食指尖距离在 [0.04,0.10]m 上归一化，EgoDex 惯例） |
| 29 | right_gripper | 右手同上 |
| 30–41 | left/right_hand | **空（mask=0），预留给 retarget 后的手部 DoF** |
| 42–45 | waist | 空 |
| 46–47 | head | 头部 yaw_rel+pitch（相对躯干朝向，弧度） |
| 48–50 | mobile | 空 |
| 51–54 | reserved | 空 |

- `mask_state` / `mask_action`（各 55 维 float 0/1）标记哪些维有效，训练时必须用
  它们屏蔽空槽位。
- **坐标系**：所有 55D 有效槽位在**逐帧躯干偏航系**（z 轴对齐重力，yaw=躯干朝向）。
  body 子集：原点=pelvis、yaw=肩线方向；非 body 子集：原点=头部、yaw=头朝向代理
  （实测代理误差中位 5.8°，详见 info.json `custom.coordinate_frames`）。
- **四元数**：55D 输出统一 **xyzw**；而透传列 `observation.pose.*` 保持源格式
  **wxyz**（见第 3 节），两边不一样，取用时注意。
- **action 约定**：`action[t] = state[t+1]`（最后一帧复制自身），经由
  `action.intermediate.*` 列可自查。
- `reliability`（55 维）目前恒 1，预留给后续标注。

### 任务/子任务标注

- `task_index` → `meta/tasks.parquet`（index 即任务文本）
- `subtask_index`（逐帧 int64）→ `meta/subtasks.parquet`；**-1 = 未标注帧**
- `meta/episodes.jsonl` 的 `action_config` 是 sidecar（episode 相对帧区间 +
  action_text + skill），与逐帧列同源可互验

### 时间轴（重要）

视频一律按 30fps 打包（LeRobot v3 每数据集单一 fps）。源数据约 21% 的 episode
原生 25fps，这些集的容器时间轴被压缩 25/30。**帧索引、action_config 帧区间
永远精确**；真实墙钟时间用 `source_timestamp` 列和 episodes.jsonl 里的
`source_fps` / `source_duration_s`。

## 3. 透传列（保留未处理的 mcap 原始数据）——取用指南

这些列是 mcap 原始流的**逐帧全保真**拷贝（float32），我们只做了时间对齐
（以 `/pose/head` 时钟为帧基准，最近邻对齐），**没有做任何坐标变换或滤波**。
词表：`observation.pose.*` 每关节 7 维 = pos xyz + quat **wxyz**，世界系。

| 列 | 形状 | 内容 |
|---|---|---|
| `observation.pose.left_hand` / `right_hand` | **(147,)** = 21 关节 × 7 | 全手部骨架，世界系。**retarget 就用它** |
| `observation.pose.body` | (154,) = 22 × 7 | 全身骨架（仅 body 子集）；末 2 个是腕部占位副本 |
| `observation.pose.upper_body` / `lower_body` | (98,)/(56,) = 14×7 / 8×7 | 上/下半身子集 |
| `observation.pose.head` | (7,) | 头部位姿（帧时钟基准流） |
| `observation.pose.headcam` / `right_eye_cam` | (7,) | 相机位姿 |
| `observation.calibration.<cam>.K/P/R` | (9,)/(12,)/(9,) | 内参，**K/P 已同步 ×0.5 适配 960×728**，R 不变 |
| `observation.extrinsics.<cam>` | (7,) | 外参 pos xyz + quat **xyzw**（注意与 pose.* 不同！） |
| `annotation.bad_frame.pose_*` | bool | 该帧该组姿态是否坏帧 |
| `annotation.bad_frame.pose_*.problem_types` | string | JSON 字符串的坏帧原因列表 |
| `source_timestamp` | float64 | 真实采集时钟（秒，episode 相对） |

手部 21 关节顺序（`info.json custom.hand_joint_order` 也有）：
`0=wrist, 1-4=thumb(cmc,mcp,ip,tip), 5-8=index(mcp,pip,dip,tip), 9-12=middle,
13-16=ring, 17-20=pinky`。**关节 0=wrist 就是 eef 槽的来源**；
关节 4/8（拇指尖/食指尖）是 gripper 捏合度的来源。

body 22 关节顺序见 `custom.body_joint_order`（0=pelvis, 16/17=左右肩——
躯干 yaw 就是从肩线算的）。

## 4. 手部 retarget → 回填 55D 的操作指引（已知场景）

1. **取数**：直接从 data parquet 读 `observation.pose.{left,right}_hand`
   （147 维，世界系，wxyz），无需回原始 mcap。若还需原始视频内参做投影，
   用 `observation.calibration.*`（已适配 960×728）。
2. **坐标系选择**：透传列在世界系（人物大移动时会整体漂移）；55D 槽位在
   逐帧躯干偏航系。回填前建议用与 staging 相同的变换把 retarget 结果转进
   躯干系（参考 `hf_tools/egostandard_staging.py` 的 `_to_torso()` 与
   torso_yaw 计算；每帧的躯干原点/yaw 可由 `observation.pose.body` 的
   pelvis+肩线重算，非 body 子集用 head 代理）。
3. **回填位置**：
   - 手部 DoF → `left_hand` 槽 **30:36** / `right_hand` 槽 **36:42**（各 6 维，
     如 6 维不够需要扩展 schema，请先与本仓库 Owner 对齐）
   - retarget 后的夹爪 → **覆盖** `left/right_gripper`（28/29）里现有的
     捏合度占位值（现有值是拇指-食指距离的启发式，不是 retarget 结果）
4. **回填后必须同步**：把对应维度的 `mask_state`/`mask_action` 置 1，
   并更新 `meta/info.json` 的 `custom` 记录（怎么 retarget 的、坐标系、
   单位），否则下游无法区分"真空"与"未填"。

## 5. 已知数据缺陷与注意点

### mcap 逐字段对账（2026-09-07 全量审计）

源 mcap 的**每个 topic、每个字段**与输出的对应关系：

| mcap 内容 | 去向 |
|---|---|
| `/pose/*`（全部 transforms） | 透传列 `observation.pose.*`，逐帧全保真 |
| `/sensor/camera/*/intrinsic` (K/R/P) | `observation.calibration.*`（K/P 随降分辨率 ×0.5） |
| `/sensor/camera/*/extrinsic` | `observation.extrinsics.*` |
| `/sensor/camera/*/video` | 960×728 h264 重编码（**有损，有意为之**） |
| `/annotation/bad_frame/*` (is_bad, problem_type) | `annotation.bad_frame.*` 列 |
| `/annotation/semantic_segments` 的 subtask_description / end_time / task_description | subtask_index 列 + subtasks.parquet + episodes.jsonl |
| `/annotation/semantic_segments` 的 **skill** | **未保留**——proto 里有此 repeated 字段，但 60 个随机任务抽样验证**源数据全部为空**，无实际损失 |
| `/annotation/semantic_segments` 的 **start_time** | **未保留**——抽样验证段间恒连续且首段恒从 0 开始，与 end_time 冗余 |
| `/session/metadata`（设备/采集员/场景/墙钟时间） | episodes.jsonl 的 `session_metadata` 字段，完整保留 |
| 各 topic 消息的精确 log_time | 仅 `/pose/head` 时钟进入 `source_timestamp`，其余流最近邻对齐到该时钟（亚帧级对齐抖动丢失） |
| 深度相机 | session_metadata 里登记了 head_depth_camera 设备，但 mcap 中**本就没有** depth 数据流，非转换丢失 |

- `EgoStandard-body__Fur_Reversal` 第 0 集：源 mcap 右目视频流仅 731/14387 帧
  （录制缺陷），该 task 未做视频合并，使用时注意 ep0 右目在 24s 后无画面。
- 非 body 子集 head 槽的 yaw_rel 恒为 0（头朝向即躯干代理朝向，已知退化），
  pitch 仍有效。
- 视频从 1920×1456 重编码到 960×728（h264 crf18，精确减半，内参已同步缩放）。
  原始分辨率只在源 mcap 里。
- 早期部分 task 的视频可能是 per-episode 小文件（合并回补分批进行），
  索引列均已正确指向，加载无差别。

## 6. 环境坑（lerobot 环境）

- **不要**在 import lerobot 之前 import pyarrow/pandas（torchcodec 动态库
  会加载失败）；视频解码用 PyAV 绕过（`ds.hf_dataset` 取数据列不触发解码）。
- 参考实现：`hf_tools/egostandard_55d_official_load.py`（lerobot env：
  `/home/z00827204/miniconda3/envs/lerobot/bin/python`）。
