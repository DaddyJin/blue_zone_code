# InternData-A1 (Franka sim) → 统一 55D LeRobot v3.0 转换

> 本文档记录 `InternData-A1` demo 数据集(`close_the_laptop`)转换到团队统一格式
> (见 `../standard_format/README.md`)的完整流程、决策依据与验证结果。
> 转换日期:2026-10-15/16。工具链复用并扩展自 GM100 / lingbot 项目。

---

## 1. 概述

| 项 | 内容 |
|---|---|
| 源数据 | `/data5T/liujin/dataset/InternData-A1/sim_updated_lerobotv30/articulation_tasks/franka/close_the_laptop.tar.gz`(压缩包,内含 **LeRobot v3.0** 布局数据集) |
| 目标格式 | 团队统一格式:LeRobot **v3.0** 官方布局 + 55D 统一动作列(见 `../standard_format/README.md`) |
| 产物 | 本目录下 `processed_dataset/close_the_laptop/`(每个压缩包内数据集一个独立输出,保持源名) |
| 机器人 | **Franka**(仿真),**单臂 7-DOF** + 平行夹爪 |
| 相机 | `head`(640×360)、`hand`(640×480),源为 **AV1** 编码,30fps |
| 规模 | **578 episodes / 140,157 frames / 11 视频文件** |

**输入为压缩包**:大批量场景下输入是若干 `*.tar.gz`,先跑 `intern_extract.py` 解压
(幂等、可断点续跑),再转换。

---

## 2. 源数据诊断(转换前实际读取确认)

### 2.1 数据布局

`close_the_laptop.tar.gz` 解压后是一个 **LeRobot v3.0** 数据集(非 v2.1):

```
close_the_laptop/
├── data/chunk-000/file-000.parquet      # 单文件,140,157 行,29 列
├── videos/images.rgb.head/chunk-000/file-000.mp4        # 全部 578 集打包,4671.9s,AV1
├── videos/images.rgb.hand/chunk-000/file-{000,001}.mp4  # 2977.8s + 1694.1s,AV1
└── meta/{info.json, stats.json, tasks.parquet, episodes/chunk-000/file-000.parquet}
```

**与团队标准的差距**(需要处理):
1. 视频是 **AV1**、且**打包严重超 960s 上限**(4671.9s / 2977.8s / 1694.1s)→ 需转码 h264 + 按 episode 边界重打包
2. 相机键名 `images.rgb.*` → 规范要求 `observation.images.*`
3. 缺 55D 列 / `subtask_index` / episodes.jsonl / modality.json

### 2.2 源数据提供的通道(episode parquet 的 29 列)

| 类别 | 列 | 说明 |
|---|---|---|
| 关节 | `states.joint.position` (7) | 7-DOF 关节角(弧度) |
| 夹爪 | `states.gripper.position` (标量) | 夹爪开合,米,[0, 0.08](Franka 行程) |
| 位姿 | `states.gripper.pose` (6) | **法兰**:位置3 + 欧拉角3(URDF rpy) |
| 位姿 | `states.ee_to_{armbase,robot}_pose` (7) | 法兰:位置3 + 四元数4(**缺陷,见 §7.1**) |
| 位姿 | `states.tcp_to_{armbase,robot}_pose` (7) | TCP(法兰前 0.095m):位置3 + 四元数4(同缺陷) |
| 场景 | `states.robot_to_env_pose` (7) | 机器人在环境系的位姿 |
| 标定 | `{head,hand}_camera_intrinsics` (4) / `..._to_robot_extrinsics` (7) | 相机内参/外参 |
| 动作 | `actions.*` ×7 列 | 命令目标(与 states 实测有差) |
| 主臂 | `master_actions.*` ×4 列 | 遥操作主臂信号 |

### 2.3 关键诊断结论

1. **源数据直接提供末端位姿**(与 GM100 相反,无需 URDF/找模型)——但需甄别:
   - `states.ee_to_*` / `tcp_to_*` 的**四元数部分是源数据的缺陷编码**(穷举 all 常规约定/置换/符号组合均无法还原真实姿态,详见 §7.1);
   - `states.gripper.pose` 的**欧拉角是干净的真值**,与标准 Franka 运动学 **0.0000°** 全量一致;
2. 原生 30fps(无帧率归一问题);`action.*` 是命令目标,不是"下一帧 state";
3. 无子任务标注(`subtask_index` 恒 -1)。

---

## 3. 55D 槽位映射(单臂 → 左臂槽)

55D 布局参见 `../standard_format/README.md` §3.2。本数据集填充面:

| 槽位 | 维度 | 填充内容 | mask |
|---|---|---|---|
| **left_arm_joint** | 0:7 | 7 关节角(弧度,7-DOF 正好填满) | 1 |
| right_arm_joint | 7:14 | 空(单臂) | **0** |
| **left_eef** | 14:21 | **法兰位姿**:位置(m, 基座系) + 四元数 xyzw | 1 |
| right_eef | 21:28 | 空 | **0** |
| **left_gripper** | 28 | 夹爪原始开合值(m,[0, 0.08],不归一化) | 1 |
| 29 / 30:55 | — | 空(hand/waist/head/mobile/reserved) | **0** |

即:**15 维有真实数据**(7 关节 + 7 eef + 1 夹爪),其余 40 维 mask=0。

### 3.1 eef 取"法兰位姿"的依据(与规范先例一致的约定)

- **物理点**:法兰(工具安装面),不是 TCP(抓取中心,在其前方 0.095m)——
  与 egosuite(腕部)、agibot(end 位姿)、GM100/lingbot(link6 腕部法兰)的先例语义一致;
- **来源**:`states.gripper.pose` 的**位置 + 欧拉角→四元数(xyzw)换算**
  (scipy `from_euler("xyz")`,URDF rpy 固定轴约定);
- **验证**:与标准 Franka DH 正运动学(`intern_fk.py`,仅用关节角)**全量 140,157 帧
  偏差 ≤ 0.034°**;位置残差 ≤ 9.5e-08 m —— 两条独立路径互相印证;
- 坐标系 = **机器人基座系**(`armbase`;与标准 Franka DH 基座零偏移,已实测)。

### 3.2 action 语义

`action_unified[t] = state_unified[t+1]`(末帧复制自己,**严格在 episode 内**)。
源 `actions.*`(命令目标)与 `master_actions.*`(主臂)全部**原样进透传列**。

---

## 4. 输出结构

```
InternData-A1-preprocess/
├── intern_extract.py     # ① 批量解压压缩包(幂等,§9)
├── intern_convert.py     # ② 转换(含视频转码+打包)
├── intern_fk.py          # Franka DH 正运动学(校验用,§8)
├── intern_validate.py    # ③ 校验
├── README.md             # 本文档
└── processed_dataset/
    └── close_the_laptop/
        ├── data/chunk-000/file-000.parquet      # 33 列(见 §4.1),zstd,76.1 MB
        ├── videos/observation.images.{head,hand}/chunk-000/file-*.mp4
        │                                        # 11 个 h264 文件(见 §5)
        ├── meta/
        │   ├── info.json                        # v3.0,fps=30,features+custom
        │   ├── tasks.parquet                    # 'Close the laptop'
        │   ├── subtasks.parquet                 # 空表(源无子任务标注)
        │   ├── episodes.jsonl                   # action_config=[]、source_fps/duration
        │   ├── episodes/chunk-000/file-000.parquet  # episode 索引(含逐相机定位列)
        │   └── modality.json                    # 55D 映射说明
        └── qc_report.json                       # L1 QC 摘要
```

### 4.1 data parquet 列(33 列 = 10 规范 + 22 透传 + index)

| # | 列 | 类型 | 说明 |
|---|---|---|---|
| 1 | `frame_index` | int64 | episode 内帧号,0 起 |
| 2 | `episode_index` | int64 | episode 编号 |
| 3 | `timestamp` | float64 | = frame_index/30(与样例实态一致) |
| 4 | `task_index` | int64 | 恒 0 |
| 5 | `state_unified` | float32[55] | 见 §3 |
| 6 | `action_unified` | float32[55] | = 下一帧 state(末帧复制) |
| 7 | `mask_state` | float32[55] | 0/1,空槽=0 |
| 8 | `mask_action` | float32[55] | 同上 |
| 9 | `reliability` | float32[55] | 当前实现恒等于 mask |
| 10 | `subtask_index` | int64 | 恒 **-1**(源无标注,规范要求的注册列) |
| 11-32 | 22 个透传列 | float32 | 见下 |
| 33 | `index` | int64 | 全局行号 0..N-1 |

**透传列**(原样保留,含 4 相机内外参、`states.gripper.pose`、ee/tcp 位姿 ×4、
`robot_to_env_pose`、`actions.*` ×7、`master_actions.*` ×4):

- 1 维特征按规范以**标量**存储(`states.gripper.position`、两组 `.openness`);
- 已映射进 55D 的 `states.joint.position` / `states.gripper.position` **不重复透传**(对齐前例)。

### 4.2 episodes.jsonl

每行:`{episode_index, tasks, length, action_config: [], source_fps: 30.0, source_duration_s}`。

---

## 5. 视频规范(方案 B:转码 + ≤960s 打包,与规范生产终态一致)

源视频是 AV1 且打包超长(最大 4671.9s),按规范 §5 处理:

| 项 | 值 |
|---|---|
| 编码 | **libx264,crf=18,preset=fast,yuv420p,+faststart**(规范 §5 指定参数) |
| 分辨率 | **保留源生** 640×360(head)/ 640×480(hand)(960×728 为 egosuite 专属缩放) |
| 打包 | **按 episode 顺序贪心装箱,每文件 ≤960s(28800 帧),只在 episode 边界切** |
| 实现 | **一遍 ffmpeg 完成转码+切分**(`-f segment -segment_frames`,切点帧数零误差),随后逐文件流拷贝加 `+faststart`(segment muxer 不传递 movflags,已实测) |
| 输出 | head 5 个 + hand 6 个 = **11 个文件**;head file-000..004 / hand file-000..005 |
| 实际大小 | head ~49-55MB/文件;hand ~10-142MB/文件;**合计 ~888MB**(单文件最大 142MB,满足 ≤500MB 轮转) |
| 索引 | `from/to_timestamp` = **文件内累计时长**(每文件从 0 开始,与样例语义一致) |

`info.json`:`video_files_size_in_mb = 500`、`data_files_size_in_mb = 100`(规范 §4.1
"必须与 §5 轮转一致")。

---

## 6. 已知数据特征(非缺陷,如实记录)

1. **源 ee/tcp 四元数为缺陷编码**(见 §7.1):原样保留在透传列,**未用于** 55D eef;
   55D eef 用 `gripper.pose`(位置+欧拉角)且经 FK 全量核验;
2. **`actions.gripper.openness` 全为 0**(退化列,源数据如此;原样透传);
3. **`robot_to_env_pose` 有轻微波动**(位置范围 x≤0.05 / z≤0.20m;源数据如此);
4. **无子任务标注**:`subtask_index` 恒 -1、`subtasks.parquet` 空表(保留 schema);
5. **lerobot ≤0.4.x 的 `__getitem__` 缺陷**(非数据问题,见 §7.2):本地取帧需 workaround。

---

## 7. 重要发现与兼容性说明

### 7.1 源数据 ee/tcp 四元数是缺陷编码

对 `states.ee_to_*` / `tcp_to_*` 的四元数做了系统性检验(刚体挂载一致性、TCP 偏移
恒定性、24 欧拉约定 × 4! 置换 × 16 符号翻转 × 全局符号的穷举搜索),**没有任何标准
编码约定能还原真实姿态**;而同行的位置、欧拉角字段全部精确。判定为源导出器的缺陷。

**处理**:原样透传 + `info.json.custom.coordinate_frames` 注明;55D eef 改用
`gripper.pose`(且经标准 Franka FK 全量核验 0.034°)。

### 7.2 lerobot 0.4.4 的 `__getitem__` 缺 `subtask_index == -1` 保护

lerobot 0.4.4 的 `__getitem__` 末尾:

```python
if "subtask_index" in self.features and self.meta.subtasks is not None:
    item["subtask"] = self.meta.subtasks.iloc[subtask_idx].name
```

对 `subtask_index == -1` 的帧,`iloc[-1]` 在**空 subtasks 表**上抛 IndexError——
这是 0.4.4 的库缺陷(**任何**符合团队格式、无子任务标注的数据集都会触发;
本团队规范 §7#4 明确要求"统计 -1 占比",即标准环境 lerobot 0.5.0 已处理 -1)。

**本地 workaround(一行)**:加载后设置 `ds.meta.subtasks = None`(跳过该查找;
语义等价于 0.5.0 对 -1 帧的处理)。`intern_validate.py` 已内置该 workaround 并
完成真实视频解码取帧验证。

---

## 8. 验证结果(全部通过)

`intern_validate.py` 全量校验(**ALL CHECKS PASSED,0 failures**;1 条 warning 为
§7.2 的库缺陷说明):

| # | 检查 | 结果 |
|---|---|---|
| 1 | info.json:v3.0 / fps=30 / features 全注册 / video 500MB / data 100MB | ✅ |
| 2 | data parquet 列序 33 列与规范一致(逐列 dtype 核对) | ✅ |
| 3 | mask/reliability 模式:active=[0:7, 14:21, 28],其余恒 0(逐集核验) | ✅ |
| 4 | 空槽全 0;eef 四元数模=1(逐集核验) | ✅ |
| 5 | **FK(关节角) == 55D eef:max 位置 9.5e-08 m,max 角度 0.034°(全量 140,157 帧)** | ✅ |
| 6 | `action[t] == state[t+1]`,末帧复制自身(逐集核验) | ✅ |
| 7 | 帧数三级对账(Σlength == 行数 == total_frames == 140,157) | ✅ |
| 8 | episodes 索引 + 逐相机 from/to 文件内累计链(head 5 文件 / hand 6 文件) | ✅ |
| 9 | 视频:**全部 11 文件 ≤960s**、帧数逐一核对、逐文件真实解码、+faststart;**drift ≤ 0.002s ≤ 0.5s**(规范 §7#1);容器解码时间戳(dts)均匀性 0.000% ≤ 2%(§7#5 类比,stts 语义) | ✅ |
| 10 | 词表:tasks 1 行;subtasks 空表 schema 正确 | ✅ |
| 11 | **LeRobotDataset 官方加载 len=140,157;随机 10 帧(含文件边界)真实解码取帧** | ✅(含 §7.2 workaround) |

参考数据:eef 位置范围 x[0.214, 0.681] y[-0.447, 0.460] z[0.033, 0.660] m;
速度参考:转码 ~11 分钟,全量校验 ~1 分钟。

---

## 9. 执行流程与批量指南

```
源:压缩包                                输出:v3.0 + 55D 数据集
  .../franka/close_the_laptop.tar.gz        processed_dataset/close_the_laptop/
        │ ① intern_extract.py 解压+结构校验(幂等)
        ▼
    close_the_laptop/(v3.0 源)
        │ ② intern_convert.py 55D 构造(关节/法兰eef/夹爪)→ 22 透传列 →
        │                     数据轮转 → 视频转码+≤960s 打包(+faststart)→ meta
        ▼
        │ ③ intern_validate.py 全量校验(§8,含 FK 交叉核对与全量视频核对)
        ▼
      交付(ALL CHECKS PASSED 才算完成)
```

**单数据集**:

```bash
conda activate cosmos-framework
python intern_extract.py                              # [SKIP] 已解压
python intern_convert.py  --task close_the_laptop     # ~12 分钟(视频转码为主)
python intern_validate.py --task close_the_laptop     # 必须 ALL CHECKS PASSED
python intern_fk.py                                   # FK 模块自检(秒级)
```

**大批量(一个文件夹下多个压缩包) — 逐步命令**

场景:压缩包(每个含一个数据集目录,如 `new_task.tar.gz` → `new_task/`)位于数据根下
(可嵌套子目录,如 `.../articulation_tasks/franka/*.tar.gz`)。

```bash
# ── 第 0 步:环境(每次会话一次) ──────────────────────────────────
cd /data5T/liujin/code/data_preprocess/InternData-A1-preprocess
CONDA_HOME="/data5T/liujin/software/miniconda3"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate cosmos-framework
python intern_fk.py                        # FK 自检(秒级)

# ── 第 1 步:解压全部压缩包(幂等;递归扫描 *.tar.gz/*.tgz) ────────
python intern_extract.py --src-root /data5T/liujin/dataset/InternData-A1 --continue-on-error
#   每包一行:[OK] 新解压 / [SKIP] 已解压;末尾 EXTRACT_DONE ok=.. skip=.. failed=0
#   有 [FAIL]:看日志原因,修好后原样重跑本命令(成功的自动 SKIP)

# ── 第 2 步:pilot — 先转一个数据集并校验(推荐) ─────────────────
python intern_convert.py  --task <新数据集名>     # 如 new_task
python intern_validate.py --task <新数据集名>     # 必须 0 failures 再进下一步

# ── 第 3 步:全量转换(后台 + 断点续跑) ───────────────────────────
setsid nohup python intern_convert.py --skip-existing --continue-on-error \
    > convert.log 2>&1 &
tail -f convert.log
#   每数据集一行:[OK]/[SKIP]/[FAIL];末尾 TASK_DONE ok=.. skipped=.. failed=..
#   若中断/崩溃:原样重跑同一条命令 —— 已完成的 [SKIP],自动接着转
#   失败清单在日志末尾 FAILED_TASKS: xxx;修好原因后重跑本命令只重做失败的

# ── 第 4 步:全量校验 ────────────────────────────────────────────
python intern_validate.py --all > validate.log 2>&1
tail -1 validate.log
#   期望:RESULT: ALL CHECKS PASSED (.. warnings, N tasks)
#   若失败:日志末尾 FAILED_TASKS: xxx;重转这些数据集后重校验

# ── 第 5 步:核对 ────────────────────────────────────────────────
grep -c "^\[OK\]" convert.log              # [OK]+[SKIP] 应 = 数据集总数
grep "FAILED_TASKS" convert.log            # 应无输出
ls processed_dataset | wc -l               # 产出目录数
```

**批量注意**:

- 解压默认在压缩包**旁边**生成数据集目录(`--dest` 可指定统一解压根);
- 输出名 = 数据集目录名(保持源名);不同压缩包内**同名目录**会冲突(extract 跳过/convert 报错提示);
- 转换**逐个串行**(视频转码是 CPU 密集任务);同机不要并发跑多个 convert;
- 时间/空间参考(按 close_the_laptop 规模:578 集/14 万帧):转换 ~13 分钟
  (视频转码为主)、校验 ~1 分钟;磁盘 ~1GB/数据集(data ~76MB + 视频 ~0.9GB,
  随内容码率浮动);data 轮转 ~100MB,视频按 ≤960s 打包(实测单文件 ≤142MB);
- lerobot 0.4.4 本地取帧需 workaround(§7.2),校验脚本已内置,无需手工处理。

**本地加载取帧(lerobot 0.4.4 需 workaround,§7.2)**:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/close_the_laptop', root='processed_dataset/close_the_laptop')
ds.meta.subtasks = None        # 0.4.4 quirk workaround(0.5.0 不需要)
item = ds[100000]              # 真实视频解码:observation.images.{head,hand}
```

**迁移到新任务/新环境的自改点**:同机器人(同列名)的新压缩包**无需改代码**
(解压 → convert 递归发现 → 校验);若列名/机器人不同,改 `intern_convert.py`
顶部的源列名与 `build_episode_table()` 映射(参考 GM100 项目的 §7.5)。

---

## 10. 决策记录(为什么这么做)

| 决策 | 结论 | 依据 |
|---|---|---|
| eef 物理点 | **法兰**(非 TCP) | 与 egosuite/agibot/GM100 先例一致;TCP 信息在透传列 |
| eef 来源 | `gripper.pose` 位置+欧拉角换算(主),标准 Franka FK 做独立核验 | 源四元数缺陷(§7.1);两路互证 0.034° |
| 单臂槽位 | 左臂 0:7 + 左 eef 14:21 + 左 gripper 28;右臂 mask=0 | 用户确认;规范"有什么填什么" |
| action 语义 | `action[t]=state[t+1]`,命令目标/主臂进透传 | 规范 §3.2 |
| gripper 量纲 | 原始值 [0, 0.08] 不归一化 | 与 agibot/GM100 一致;归一化是下游职责 |
| 视频方案 | **转码+≤960s 打包(方案 B)**,crf18/preset fast/+faststart | 规范 §5 生产终态(方案逐集为过渡态);规范指定编码参数 |
| 视频切分实现 | 一遍 ffmpeg(segment)+ 逐文件 faststart 后处理 | segment muxer 不传递 movflags(实测);帧数零误差(实测) |
| 分辨率 | 保留源生 640×360/640×480 | 960×728 系 egosuite 专属缩放;硬门槛全满足 |
| timestamp dtype | float64 | 与两个官方样例实态一致(规范表写 float32) |
| 子任务 | 恒 -1,空词表 | 源无标注;0.4.4 quirk 见 §7.2 |
| 轮转口径 | data ~100MB 磁盘大小(自校准压缩比) | lerobot 同口径;本数据集 76MB 单文件未触发 |
