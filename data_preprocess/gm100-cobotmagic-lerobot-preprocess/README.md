# GM100 (songling_selfcollect) → 统一 55D LeRobot v3.0 转换

> 本文档记录 `gm100-cobotmagic-lerobot-demo` 数据集转换到团队统一格式
> (见 `../standard_format/README.md`) 的完整流程、决策依据与验证结果。
> 转换日期:2026-10-15(同日完成 eef 正运动学回填,见 §3.2)。

---

## 目录

- [1. 概述](#1-概述)
- [2. 源数据诊断(转换前实际读取确认)](#2-源数据诊断转换前实际读取确认)
- [3. 55D 槽位映射(本数据集的实际填充)](#3-55d-槽位映射本数据集的实际填充)
  - [3.1 为什么关节槽是 7 宽而不是 6](#31-为什么关节槽是-7-宽而不是-6)
  - [3.2 eef 槽:URDF 正运动学回填](#32-eef-槽urdf-正运动学回填)
  - [3.3 action 语义:下一帧 state(不是命令目标)](#33-action-语义下一帧-state不是命令目标)
- [4. 输出结构](#4-输出结构)
  - [4.1 data parquet 列(16 列 = 11 规范 + 5 透传)](#41-data-parquet-列16-列--11-规范--5-透传)
  - [4.2 data 文件轮转](#42-data-文件轮转)
  - [4.3 视频:逐集硬链接(不合并、不重编码)](#43-视频逐集硬链接不合并不重编码)
- [5. 已知数据特征(非缺陷,如实记录)](#5-已知数据特征非缺陷如实记录)
- [6. 验证结果(全部通过)](#6-验证结果全部通过)
- [7. 执行流程与迁移指南(在新环境批量转换用)](#7-执行流程与迁移指南在新环境批量转换用)
  - [7.0 流程总览](#70-流程总览)
  - [7.1 依赖](#71-依赖)
  - [7.2 迁移清单(拷到新环境的最小文件集)](#72-迁移清单拷到新环境的最小文件集)
  - [7.3 执行步骤(以本数据集为例)](#73-执行步骤以本数据集为例)
  - [7.4 参数说明](#74-参数说明)
  - [7.4.1 进度与并行(大批量)](#741-进度与并行大批量)
  - [7.5 迁移到新数据集时改什么](#75-迁移到新数据集时改什么)
  - [7.6 批量转换注意事项](#76-批量转换注意事项)
  - [7.7 大批量完整示例(如:A 下有 1000 个 task)](#77-大批量完整示例如a-下有-1000-个-task)
- [8. 决策记录(为什么这么做)](#8-决策记录为什么这么做)

## 1. 概述

| 项 | 内容 |
|---|---|
| 源数据 | `/data5T/liujin/dataset/gm100-cobotmagic-lerobot-demo`(lerobot **v2.1**,per-episode 布局) |
| 目标格式 | 团队统一格式:LeRobot **v3.0** 官方布局 + 55D 统一动作列 |
| 产物 | 本目录下 `processed_dataset/task_00001/`、`task_00002/`、`task_00003/`(每个 task 一个独立数据集) |
| 机器人 | `songling_selfcollect`,双臂 **6-DOF × 2** + 左右夹爪 |
| 相机 | `camera_top`、`camera_wrist_left`、`camera_wrist_right`,640×480 @30fps h264 |
| 总规模 | **440 episodes / 533,571 frames / 1,320 视频** |

**产物可直接被 `LeRobotDataset` 加载**(已在 lerobot 0.4.4 实测,见 §6)。

---

## 2. 源数据诊断(转换前实际读取确认)

```
gm100-cobotmagic-lerobot-demo/
├── task_00001/  hit-ball-with-gripper-to-score-goal                    130 eps /  64001 帧
├── task_00002/  cut-thin-cucumber-slice-with-two-arms-and-return-knife 130 eps / 250083 帧
└── task_00003/  dip-stamp-in-ink-and-stamp                             180 eps / 219487 帧
```

每个 task 是一个 **lerobot v2.1** 数据集:

- `data/chunk-000/episode_XXXXXX.parquet`(每集一个文件,428~1007 帧不等)
- `videos/chunk-000/<key>/episode_XXXXXX.mp4`(每集每相机一个文件,~16-34s)
- `meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, quality_check.jsonl}`

**源数据提供的通道**(episode parquet 的 12 列):

| 源列 | 宽度 | 说明 |
|---|---|---|
| `observation.state.arm.position` | 12 | 左臂6关节 + 右臂6关节(**弧度**) |
| `observation.state.arm.velocity` | 12 | 关节速度 |
| `observation.state.arm.effort` | 12 | 关节力矩 |
| `observation.state.effector.position` | 2 | 左/右夹爪开合(**原始量**,范围见 §5) |
| `observation.state.effector.effort` | 2 | 夹爪力矩 |
| `action.arm.position` | 12 | **命令目标**(下发值,非实际到达值) |
| `action.effector.position` | 2 | 夹爪命令目标 |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | - | 索引列 |

**关键诊断结论**:

1. **无末端位姿(eef)**:只有关节角,没有 `end.position`/`orientation`,数据集内也无 URDF;
2. **无子任务标注**:`episodes.jsonl` 只有 `{episode_index, tasks, length}`,无 `action_config`/`skill`;
3. 原生 30fps(无帧率归一问题);`action.*` 是命令目标,**不是**标准格式约定的"下一帧 state"。

---

## 3. 55D 槽位映射(本数据集的实际填充)

55D 布局参见 `../standard_format/README.md` §3.2。本数据集的填充面:

| 槽位 | 维度 | 填充内容 | mask |
|---|---|---|---|
| left_arm_joint | **0:6** | 左臂 6 关节角(弧度) | 1 |
| (空) | 6 | 6-DOF 臂在 7 宽槽中的空位 | **0** |
| right_arm_joint | **7:13** | 右臂 6 关节角(弧度) | 1 |
| (空) | 13 | 同上 | **0** |
| left/right_eef | **14:28** | **FK 正解**:腕部法兰(link6)位姿,pos(3)+quat xyzw(4),机器人基座系 | 1 |
| left/right_gripper | **28:30** | 夹爪开合原始值(不归一化) | 1 |
| hand / waist / head / mobile / reserved | 30:55 | 空 | **0** |

即:**28 维有真实数据**(12 关节角 + 14 eef + 2 夹爪),其余 27 维 mask=0。

### 3.1 为什么关节槽是 7 宽而不是 6

55D 是**全数据集统一的并集 schema**,手臂关节槽固定 7 宽/臂(参照 7-DOF 的 agibot)。
GM100 是 6-DOF,所以"**填 6 + 空 1**":左臂占 0:6、右臂占 7:13,第 6/13 维置 0 且 mask=0。
这是规范"有什么填什么、缺的置空"的标准做法(对照:egosuite 无关节角,0:14 整段 mask=0)。

### 3.2 eef 槽:URDF 正运动学回填

源数据只有关节角、没有末端位姿。用 Piper 臂 URDF(`urdf/aloha_tracer2_dabai_dark.urdf`,
来源与选择依据见 `urdf/README.md`)做**正运动学(FK)**,把关节角换算成末端位姿,
回填 14:28 并置 mask=1。实现:`gm100_fk.py`。

- **eef 定义**:腕部法兰(URDF `link6` 帧)的位姿;映射 `left_arm_0..5 → fl_joint1..6`、
  `right_arm_0..5 → fr_joint1..6`
- **坐标系**:URDF 根帧 `footprint`(与臂安装面 `body_Link` 重合)= 机器人基座系;
  位置单位米,四元数 **xyzw**(55D 约定)
- **假设(待硬件确认)**:GM100 实体臂 = Piper 臂;若型号不符,需换 URDF 重算
- **未纳入**:夹爪 TCP 偏移(用 link6 腕部帧;如需 TCP 可加常量偏移后重算)

**对齐验证证据**(转换前执行;URDF 关节限位 vs 数据实际分布):

| 关节 | URDF 限位 | GM100 实测范围 | 结论 |
|---|---|---|---|
| J2 | **[0°, +180°]**(仅允许正向) | [-0.2°, +151.6°](几乎纯正向) | 零位/符号一致 |
| J3 | **[-154.5°, 0°]**(仅允许负向) | [-142.7°, +0.2°](几乎纯负向) | 零位/符号一致 |
| J5 | ±70° | [-70.7°, +74.6°] | 极值紧贴限位 → 同款臂 |
| J1 / J4 / J6 | ±150° / ±105° / ±180° | 均在限内 | 一致 |

**FK 结果的独立核验**:左右 eef 距各自安装座最大 0.565 / **0.630 m**
(0.630 恰为 Piper 臂标称臂展);轨迹平滑(帧间位移 p95 < 1.3 cm);
四元数模恒为 1;左右臂 y 区间自然分离(左 y>0 / 右 y<0)。全量校验见 §6。

对照:egosuite(动捕直接测手腕位姿)与 agibot(采集时直接存 end.position)的 eef 是实测值;
GM100 的 eef 是**从关节角推算**的——两者都符合 55D 规范,来源差异已在
`info.json.custom.mapping_55d` 与本表注明。

### 3.3 action 语义:下一帧 state(不是命令目标)

规范 §3.2 约定 `action_unified[t] = state_unified[t+1]`(末帧复制自己,**严格在 episode 内 shift**)。

源数据的 `action.arm.position` 是**命令目标**(下发值),与实际到达值有实打实的差距
(实测 task_00001 ep0:指令目标 vs 下一帧实际平均差 0.0111 rad ≈0.64°,最大 0.081 rad ≈8.1°)。
两者不能混用——标准格式选"下一帧 state"是为了与具体控制器解耦。
因此:**命令目标保留为透传列,action_unified 按 shift 生成**。

---

## 4. 输出结构

```
gm100-cobotmagic-lerobot-preprocess/
├── gm100_convert.py      # 转换脚本(§7)
├── gm100_validate.py     # 校验脚本(§7)
├── gm100_fk.py           # eef 正运动学模块(§3.2)
├── _progress.py          # 进度/ETA 报告器(标准库,无额外依赖)
├── urdf/                 # Piper 臂 URDF + 来源说明(§3.2)
├── README.md             # 本文档
└── processed_dataset/    # ← 所有转换产物
    ├── task_00001/       # 每个 task 一个独立数据集
    │   ├── data/chunk-000/file-000.parquet     # 16 列(见 §4.1),zstd
    │   ├── videos/<key>/chunk-000/file-XXX.mp4 # 逐集硬链接(§4.3)
    │   ├── meta/
    │   │   ├── info.json                       # v3.0,fps=30,features+custom
    │   │   ├── tasks.parquet                   # 任务词表
    │   │   ├── subtasks.parquet                # 空表(源无子任务标注)
    │   │   ├── episodes.jsonl                  # sidecar(action_config=[])
    │   │   ├── episodes/chunk-000/file-000.parquet  # episode 索引(定位 data/videos)
    │   │   └── modality.json                   # 55D 映射说明
    │   └── qc_report.json                      # L1 QC 摘要
    ├── task_00002/ (同构)
    └── task_00003/ (同构)
```

### 4.1 data parquet 列(16 列 = 11 规范 + 5 透传)

| # | 列 | 类型 | 说明 |
|---|---|---|---|
| 1 | `frame_index` | int64 | episode 内帧号,0 起 |
| 2 | `episode_index` | int64 | episode 编号 |
| 3 | `timestamp` | float64 | = frame_index/30 |
| 4 | `task_index` | int64 | 指向 tasks.parquet(本数据集恒 0) |
| 5 | `state_unified` | float32[55] | 见 §3 |
| 6 | `action_unified` | float32[55] | = 下一帧 state(末帧复制) |
| 7 | `mask_state` | float32[55] | 0/1,空槽=0 |
| 8 | `mask_action` | float32[55] | 同上 |
| 9 | `reliability` | float32[55] | 当前实现恒等于 mask |
| 10 | `subtask_index` | int64 | 恒 **-1**(源无子任务标注) |
| 11-15 | 5 个透传列(下表) | float32[·] | 源列原样保留 |
| 16 | `index` | int64 | 全局行号 0..N-1 |

**透传列**(canonicalize 时源 parquet 中未进 55D 的列,"可以不处理,但不能丢"):

| 透传列 | 宽度 | 说明 |
|---|---|---|
| `observation.state.arm.velocity` | 12 | 关节速度 |
| `observation.state.arm.effort` | 12 | 关节力矩 |
| `observation.state.effector.effort` | 2 | 夹爪力矩 |
| `action.arm.position` | 12 | **命令目标**(与 action_unified 不同,见 §3.3) |
| `action.effector.position` | 2 | 夹爪命令目标 |

> 注:`observation.state.arm.position` 与 `observation.state.effector.position` 已被映射进 55D
> (关节槽/夹爪槽),信息无损失,**不再重复透传**(对齐 agibot 做法)。

### 4.2 data 文件轮转

按 lerobot 自身口径轮转(`data_files_size_in_mb=100`,比较**磁盘文件大小**,
见 lerobot `lerobot_dataset.py` 的 `get_file_size_in_mb` 用法)。脚本用"内存字节 ×
自校准压缩比"预估磁盘大小,仅在 episode 边界切分。

本数据集压缩比意外地高(~25x,大量 mask 列高度重复),三个 task 的数据都远小于 100MB,
因此**每个 task 各一个 data 文件**,未触发轮转:

| task | 帧数 | data 文件 | 磁盘大小 |
|---|---|---|---|
| task_00001 | 64,001 | file-000.parquet ×1 | 6.6 MB |
| task_00002 | 250,083 | file-000.parquet ×1 | 35.8 MB |
| task_00003 | 219,487 | file-000.parquet ×1 | 19.9 MB |

### 4.3 视频:逐集硬链接(不合并、不重编码)

- 布局转换:`videos/chunk-000/<key>/episode_XXXXXX.mp4` → `videos/<key>/chunk-000/file-XXX.mp4`
  (`file_index == episode_index`);
- 用 `os.link` **硬链接**,零拷贝、零新增磁盘、零画质损失(已核验 inode 相同);
- **不做 500MB 合并**:标准格式 §5 明确"切分只在 episode 边界"、per-episode 小文件
  "索引列均已正确指向,加载无差别";本数据集每集 ~16-34s,远低于 960s 硬门槛;
- `info.json` 的 `video_files_size_in_mb` 登记为实际最大单文件大小(而非照抄 500)。

---

## 5. 已知数据特征(非缺陷,如实记录)

1. **gripper 量纲与其他数据集不同**:GM100 夹爪值范围约 **[0, 0.08]**
   (task_00001 左爪 0.0047~0.0065、右爪≈0);agibot 是 ~[35, 120];egosuite 是 [0,1] 捏合度。
   各数据集槽位语义一致但量纲不同,**跨数据集训练靠下游归一化统一**——转换阶段保留原始值,不强行归一化。
2. **task_00001 夹爪基本不动**:该任务是"用夹爪击球",夹爪作为刚性拍子使用,开合量几乎恒定;
   task_00002/00003(切黄瓜/盖章)夹爪有真实开合(最大 ~0.078)。
3. **无子任务标注**:`subtask_index` 恒 -1、`subtasks.parquet` 为空表(保留 schema)。
   若后续补充标注,可原地回填(逐帧列 + 词表 + episodes.jsonl 三件套)。
4. **eef 为 FK 推算值(非实测)**:由关节角 + Piper 臂 URDF 正解得到(§3.2),
   与 egosuite/agibot 的实测 eef 来源不同;对齐验证已通过,但"实体臂 = Piper"这一前提待硬件确认。
5. 源数据集的 `episodes_stats.jsonl` / `quality_check.jsonl` 未复制到产物(非 v3.0 必需;
   源数据集仍在原路径,信息未丢失)。

---

## 6. 验证结果(全部通过)

转换后用 `gm100_validate.py` 对三个 task 全量校验,**ALL CHECKS PASSED (0 warnings)**:

| # | 检查 | 结果 |
|---|---|---|
| 1 | info.json: v3.0 / fps=30 / 全部 features 注册(含 `subtask_index` int64) | ✅ |
| 2 | data parquet 列序 16 列与规范一致;55D 列均为 fixed_size_list<float32>[55] | ✅ |
| 3 | mask/reliability 模式:active=[0..5,7..12,14..27,28,29],其余恒 0(逐集核验) | ✅ |
| 4 | 空槽(6、13、30:55)全为 0;eef 四元数模 ≈ 1(逐集核验) | ✅ |
| 5 | `action_unified[t] == state_unified[t+1]`,末帧复制自身(逐集核验) | ✅ |
| 6 | 帧数对账:Σlength == data 行数 == info.total_frames | ✅ |
| 7 | episodes 索引:`dataset_from/to_index` 连续覆盖 [0,N);视频定位列指向正确 | ✅ |
| 8 | 视频:每集每相机文件齐全;抽样 PyAV 时长与 length/30 一致(480×640@30) | ✅ |
| 9 | 词表:tasks.parquet 1 行;subtasks.parquet 空表 schema 正确 | ✅ |
| 10 | **`LeRobotDataset` 官方加载成功**(lerobot 0.4.4,`len == total_frames`) | ✅ |

数值对账:输出 55D 中关节角/夹爪与源 parquet 逐位一致(atol=1e-6);
eef(14:28)为 FK 推算值,正确性依据 §3.2 的对齐验证与独立核验。

---

## 7. 执行流程与迁移指南(在新环境批量转换用)

### 7.0 流程总览

```
源数据集(v2.1 布局)                         输出(每 task 一个 v3.0 数据集)
  <src-root>/task_XXXXX/                       <out>/task_XXXXX/
    ├─ data/chunk-000/episode_*.parquet  ①转换  ├─ data/chunk-000/file-*.parquet
    ├─ videos/chunk-000/<cam>/*.mp4             ├─ videos/<cam>/chunk-000/file-*.mp4
    └─ meta/{info.json,tasks.jsonl,...}         └─ meta/{info.json,tasks.parquet,...}
              │
              │  ① gm100_convert.py   列映射 → 55D 构造 → eef 正运动学回填 → 透传列 →
              │                       ~100MB 轮转 → 视频链接 → meta/词表/episodes 索引生成
              ▼
              │  ② gm100_validate.py  10 项全量校验(含 lerobot 官方加载软检查)
              ▼
            交付(ALL CHECKS PASSED 才算完成)
```

### 7.1 依赖

| 类别 | 需要什么 | 说明 |
|---|---|---|
| Python | ≥ 3.10 | 脚本用到了 `list[dict]` 等新式类型标注 |
| **必需包** | `pyarrow` `numpy` `scipy` | 转换与 FK 的全部依赖;`pip install pyarrow numpy scipy` |
| 可选包 | `av`(PyAV) | 校验时的视频时长抽样(软检查,缺失只报 warning,不判失败) |
| 可选包 | `lerobot` | 校验时的官方加载(软检查,同上) |
| **不依赖** | ffmpeg | 视频是**硬链接**不做转码,转换全程不需要 ffmpeg |

本机环境(cosmos-framework,含全部可选包):
```bash
CONDA_HOME="/data5T/liujin/software/miniconda3"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate cosmos-framework
```

### 7.2 迁移清单(拷到新环境的最小文件集)

```
gm100-cobotmagic-lerobot-preprocess/
├── gm100_convert.py                   # 转换脚本
├── gm100_fk.py                        # eef 正运动学模块(convert 依赖)
├── gm100_validate.py                  # 校验脚本
├── _progress.py                       # 进度/ETA 报告器(convert/validate 依赖)
└── urdf/aloha_tracer2_dabai_dark.urdf # FK 用 URDF(convert 依赖)
```

> `gm100_convert.py` 通过**脚本自身所在目录**解析 `gm100_fk.py` 与 `urdf/`
> (不依赖当前工作目录),保持上表相对位置不变即可整体迁移。

> **迁移演练已验证(2026-10-15)**:将上表 4 项拷贝到全新路径 `/tmp/gm100_migtest/` 后,
> 从无关目录(`/tmp`)用绝对脚本路径成功完成:
> ① 转换(源在 `/data5T`、输出在另一文件系统 `/tmp` → 跨文件系统自动退化为拷贝,
> 日志有 WARN 且 `info.json.custom.videos.n_copied=6` 正确记录,已核对非硬链接);
> ② 校验(迁移后的 validate 运行 → `ALL CHECKS PASSED`)。
> 结论:工具**不依赖 cwd、不依赖原机器路径**;迁到新环境后显式传 `--src-root/--out` 即可运行,
> 同文件系统走硬链接(零占用)、跨文件系统自动拷贝(按源视频总量预留磁盘)。

### 7.3 执行步骤(以本数据集为例)

```bash
cd /data5T/liujin/code/blue_zone_code/data_preprocess/gm100-cobotmagic-lerobot-preprocess

# ① pilot:先转一个 task 并立即校验(推荐在新环境首次执行)
python gm100_convert.py  --task task_00001 --workers 8
python gm100_validate.py --task task_00001 --workers 8   # 必须 ALL CHECKS PASSED 才算通过

# ② 全量:省略 --task,自动遍历源根下全部 task_* 目录
python gm100_convert.py --workers 8
python gm100_validate.py --task task_00001 task_00002 task_00003 --workers 8

# ③ 换环境/换数据:显式传路径(迁移时推荐,不依赖脚本内的默认值)
python gm100_convert.py  --src-root /path/to/src --out /path/to/out --workers 8
python gm100_validate.py --out /path/to/out --task task_00001 ... --workers 8

# ④ FK 模块自检(秒级;验证 URDF 解析、RPY/轴角约定、零位姿态)
python gm100_fk.py

# ⑤ 加载产物(可选)
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/gm100_task_00001', root='processed_dataset/task_00001')
print(len(ds), ds.meta.tasks)
"
```

### 7.4 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `convert --src-root` | `/data5T/liujin/dataset/gm100-cobotmagic-lerobot-demo` | 源根;其下每个 `task_*` 子目录是一个 v2.1 数据集 |
| `convert --out` | `本目录/processed_dataset` | 输出根;每个 task 输出到 `<out>/<task 名>/` |
| `convert --task` | 全部 `task_*` | 指定要转的 task 目录名(相对 src-root);省略则全量 |
| `convert --urdf` | `urdf/aloha_tracer2_dabai_dark.urdf` | eef 正运动学使用的 URDF |
| `convert --skip-existing` | 关 | 跳过已有 `.convert_done` 标记的 task(**大批量断点续跑用**) |
| `convert --continue-on-error` | 关 | 单个 task 失败不中断整批,结束后列出 `FAILED_TASKS` |
| `convert --workers` | 8 | **并行转换的 task 数**(1=串行);每个 worker 独立处理一个 task |
| `validate --out` | 同 convert | 待校验的输出根 |
| `validate --task` | 省略=全部 | 要校验的 task 名(可多个) |
| `validate --all` | 关 | 校验 `--out` 下**全部**数据集(大批量用) |
| `validate --workers` | 8 | **并行校验的 task 数**(1=串行) |
| `--no-progress` | 关 | 关闭进度/ETA 输出 |
| `--progress-interval` | 10 | 进度行间隔秒数(非 TTY 日志每个间隔输出一行) |

### 7.4.1 进度与并行(大批量)

转换/校验默认每 10 秒输出一行进度(帧数加权 + 已用时间 + ETA),
`--workers` 控制同时处理的 task 数:

```bash
# 1000 个 task:4 路并行转换,日志里可持续看到进度
setsid nohup python gm100_convert.py --src-root A --out B \
    --workers 8 --skip-existing --continue-on-error > convert.log 2>&1 &
tail -f convert.log
# [PROGRESS] convert 120,000/10,000,000 frames (1.2%) elapsed 00:03:10 ETA 04:12:00 ...
# [PROGRESS] convert 1,234,567/10,000,000 frames (12.3%) elapsed 00:32:10 ETA 03:49:00 | task_00042

python gm100_validate.py --out B --all --workers 8 > validate.log 2>&1
```

- **输出完全确定**:每个 task 的输出目录独立,单 task 内仍严格串行;并行只改变
  调度顺序,不改变任何列、索引、视频命名或文件划分(数据 parquet 与串行版逐字节一致,
  已在 demo 数据验证)。
- **并行度**:视频是硬链接,转换瓶颈主要是 parquet 读写与 CPU;`--workers 8`
  是默认值。HDD/单卷场景建议先实测(观察 `iostat -x 5`),再逐步调高。
- 每个 worker 内部已把 `pyarrow` CPU 线程数限制为 1,避免 N 个 worker 叠加超订。

### 7.5 迁移到新数据集时改什么

"数据集无关"的机制(55D 构造 / mask / 轮转 / meta 与词表 / episodes 索引 / 视频链接)
可直接复用;数据集相关逻辑集中在 `gm100_convert.py` 头部常量与 `build_episode_table()`:

| 要改的地方 | 位置 | 说明 |
|---|---|---|
| 源列名 | 顶部 `SRC_ARM_POS` / `SRC_EFF_POS` / `PASSTHROUGH_COLS` | 新数据集的动作/状态列可能改名 |
| 55D 槽位映射 | `build_episode_table()` 及 `LEFT_ARM_JOINT` 等常量 | 例:新数据集直接有 eef 而无关节角,则改填 eef 槽(可参考 agibot 的 modality.json 形式) |
| FK(仅"关节角→eef"需要) | `--urdf` 指向新臂 URDF;`gm100_fk.py` 中 `GM100FK` 的 `fl`/`fr` 前缀按新 URDF 关节命名调整 | 不同臂型/命名规则 |
| 相机与视频分辨率 | **无需改**:相机名自动从 `videos/chunk-000/` 探测,分辨率自动读源 `info.json` | |
| task 目录命名 | 默认匹配 `task_*`;若不符,显式传 `--task <目录名>` | |

### 7.6 批量转换注意事项

- **同文件系统要求(重要)**:视频默认用 `os.link` 硬链接,要求源与输出在**同一文件系统**。
  跨文件系统时脚本**自动退化为拷贝**(日志输出 WARN;`info.json.custom.videos.n_copied`
  记录拷贝数量),此时输出会真实占用与源视频等量的磁盘,转换耗时也会增加。
- **幂等与中断恢复**:转换对每个 task 先删除旧目录再整体重建;每个 task 完成后写
  `.convert_done` 标记(过程标记,不属交付物)。**大批量务必带 `--skip-existing` 重跑**:
  中断后在原命令上加该参数,已完成的 task 直接跳过、自动接着转(见 §7.7 示例)。
  校验是只读的,可反复执行。
- **速度参考**(本机):3 task / 53 万帧,转换约 **1 分钟**、全量校验约 **2.7 分钟**。
- **大规模批量建议**:数据量大时按 task 分批执行,长任务用 `setsid nohup` 后台运行并留存日志
  (可参考 `standard_format/docs/EGOSUITE_PIPELINE_RUNBOOK.md` 的生产做法);
  磁盘 IO 通常是瓶颈(HDD 场景),以串行为主、并行度不宜过高。
- **完成判据**:每批转换后必须跑校验,输出 `ALL CHECKS PASSED` 才算完成;
  失败时看 `[FAIL]` 行定位(`[WARN]` 为软检查,不阻塞)。

### 7.7 大批量完整示例(如:A 下有 1000 个 task)

假设源根为 `A/`(下含 1000 个 `task_*` 目录),输出根定为 `B/`:

```bash
# ── 第 0 步:环境准备(一次性) ──────────────────────────────────
conda activate <你的环境>                  # 或 pip install pyarrow numpy scipy
python -c "import pyarrow, numpy, scipy; print('deps ok')"
python gm100_fk.py                         # FK 自检(秒级,验证 URDF 可解析)

# ── 第 1 步:确认源目录结构与命名 ───────────────────────────────
ls A/ | head                               # 应是 task_00001/ task_00002/ ... 形式
ls A/task_00001/                           # 每个 task 内应有 data/ videos/ meta/

# ── 第 2 步:pilot 单 task 转换 ─────────────────────────────────
python gm100_convert.py --src-root A --out B --task task_00001
#   期望:[OK] task_00001: N eps, M frames, K videos  →  TASK_DONE ok=1

# ── 第 3 步:pilot 校验(必须 ALL CHECKS PASSED 才继续) ──────────
python gm100_validate.py --out B --task task_00001

# ── 第 4 步:全量转换(后台 + 日志;可断点续跑) ─────────────────
setsid nohup python gm100_convert.py --src-root A --out B \
    --workers 8 --skip-existing --continue-on-error > convert.log 2>&1 &
tail -f convert.log                        # 进度:[OK]/[SKIP]/[FAIL] + [PROGRESS] 行
#   若中断/崩溃:原样重跑同一条命令即可 —— 已完成的 task 显示 [SKIP],自动接着转

# ── 第 5 步:全量校验(转换完成后) ───────────────────────────────
python gm100_validate.py --out B --all --workers 8 > validate.log 2>&1
tail -1 validate.log
#   期望:RESULT: ALL CHECKS PASSED (0 warnings, 1000 tasks)
#   若失败:日志末尾有 FAILED_TASKS: task_xxx ...;重转这些 task 后再校验

# ── 第 6 步:核对结果 ───────────────────────────────────────────
grep -c "^\[OK\]" convert.log              # 成功数(应=1000 或 [OK]+[SKIP]=1000)
grep "FAILED_TASKS" convert.log            # 失败清单(应无输出)
ls B | wc -l                               # 数据集目录数(应=1000)
```

**时间/空间参考**(按 task_00001 规模实测,2026-10-19;同文件系统视频零拷贝):
- 转换:串行约 **7 秒/task**;**8 个同规模 task:57 秒 → 8 秒(`--workers 8`,约 7.1x)**;
  1000 个约 2 小时(串行)/ 15–30 分钟(8 并行,磁盘允许时)
- 校验:串行约 **19 秒/task**;**8 个 task:153 秒 → 27 秒(`--workers 8`,约 5.7x)**;
  1000 个约 5 小时(串行)/ 约 1 小时(8 并行)
- 磁盘:data parquet 为真实占用(单 task 约 6–36 MB);视频为硬链接**零占用**
  (跨文件系统退化为拷贝时,需按源视频总量预留空间)

---

## 8. 决策记录(为什么这么做)

| 决策 | 结论 | 依据 |
|---|---|---|
| action 语义 | `action[t]=state[t+1]`,命令目标进透传 | 规范 §3.2;实测命令目标与实际差最大 8.1°,不可混用 |
| eef 槽 | **FK 回填**(mask=1) | 获得 Piper 臂 URDF 后从关节角正解回填;关节顺序/零位/符号已与 URDF 限位对齐验证(§3.2) |
| 关节槽 | 6-DOF 填 6 空 1 | 槽宽 7 是全数据集上限(agibot 7-DOF);"有什么填什么" |
| gripper | 原始值,不归一化 | 与 agibot 一致;跨数据集归一化是下游职责 |
| 子任务 | 恒 -1,空词表 | 源无标注;保留 schema 便于未来回填 |
| 视频 | 逐集硬链接,不合并 | 规范允许 per-episode 文件;每集远低于 960s;零拷贝零损失 |
| 透传 | 只透传未映射的 5 列 | 已映射列(关节/夹爪)信息无损失,不重复存(对齐 agibot) |
| 轮转口径 | 磁盘大小(lerobot 同口径),自校准压缩比 | lerobot `data_files_size_in_mb` 按磁盘文件大小比较 |
