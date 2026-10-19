# lingbot GM-100 demo → 统一 55D LeRobot v3.0 转换

> 本文档记录 `lingbot-GM-100-demo` 数据集转换到团队统一格式
> (见 `../standard_format/README.md`) 的完整流程、决策依据与验证结果。
> 转换日期:2026-10-15。工具链复用自 GM100 项目
> (`../gm100-cobotmagic-lerobot-preprocess/`,见 §2 血缘关系)。

---

## 目录

- [1. 概述](#1-概述)
- [2. 源数据诊断(转换前实际读取确认)](#2-源数据诊断转换前实际读取确认)
  - [2.0 血缘关系:与 GM100 task_00001 的关系(逐字节核验)](#20-血缘关系与-gm100-task_00001-的关系逐字节核验)
  - [2.1 数据布局](#21-数据布局)
  - [2.2 与 GM100 版本不同的 6 个视频(已逐一核对)](#22-与-gm100-版本不同的-6-个视频已逐一核对)
- [3. 55D 槽位映射(与 GM100 一致)](#3-55d-槽位映射与-gm100-一致)
  - [3.1 为什么关节槽是 7 宽而不是 6](#31-为什么关节槽是-7-宽而不是-6)
  - [3.2 eef 槽:URDF 正运动学回填](#32-eef-槽urdf-正运动学回填)
  - [3.3 action 语义:下一帧 state(不是命令目标)](#33-action-语义下一帧-state不是命令目标)
- [4. 输出结构](#4-输出结构)
  - [4.0 解压(lingbot_extract.py)](#40-解压lingbot_extractpy)
  - [4.1 data parquet 列(16 列 = 11 规范 + 5 透传)](#41-data-parquet-列16-列--11-规范--5-透传)
  - [4.2 data 文件轮转](#42-data-文件轮转)
  - [4.3 视频:逐集硬链接(不合并、不重编码)](#43-视频逐集硬链接不合并不重编码)
- [5. 已知数据特征(非缺陷,如实记录)](#5-已知数据特征非缺陷如实记录)
- [6. 验证结果(全部通过)](#6-验证结果全部通过)
- [7. 执行流程与批量指南](#7-执行流程与批量指南)
  - [7.0 流程总览(压缩包 → 交付)](#70-流程总览压缩包--交付)
  - [7.1 依赖与环境](#71-依赖与环境)
  - [7.2 单数据集执行(以 BM-001 为例)](#72-单数据集执行以-bm-001-为例)
  - [7.3 大批量完整示例(假设有 N 个压缩包)](#73-大批量完整示例假设有-n-个压缩包)
  - [7.4 批量注意事项(重要)](#74-批量注意事项重要)
  - [7.4.1 进度与并行](#741-进度与并行)
  - [7.5 迁移到新压缩包/新数据集时改什么](#75-迁移到新压缩包新数据集时改什么)
  - [7.6 迁移到新环境(一个文件夹下几百个压缩包)](#76-迁移到新环境一个文件夹下几百个压缩包)
- [8. 决策记录(为什么这么做)](#8-决策记录为什么这么做)

## 1. 概述

| 项 | 内容 |
|---|---|
| 源数据 | `/data5T/liujin/dataset/lingbot-GM-100-demo/AgileX/*.tar.gz`(压缩包,内含 lerobot **v2.1** per-episode 布局数据集) |
| 目标格式 | 团队统一格式:LeRobot **v3.0** 官方布局 + 55D 统一动作列 |
| 产物 | 本目录下 `processed_dataset/BM-001/`(每个压缩包内数据集一个独立输出,保持源名) |
| 机器人 | `agilex_cobot_magic`,双臂 **6-DOF × 2** + 左右夹爪 |
| 相机 | `camera_top`、`camera_wrist_left`、`camera_wrist_right`,640×480 @30fps h264 |
| 规模 | **130 episodes / 64,001 frames / 390 视频** |

**产物可直接被 `LeRobotDataset` 加载**(已在 lerobot 0.4.4 实测,见 §6)。

**输入为压缩包**:大批量场景下输入是若干 `*.tar.gz`,先跑 `lingbot_extract.py`
解压(见 §4.0),再转换。解压幂等、可断点续跑。

---

## 2. 源数据诊断(转换前实际读取确认)

### 2.0 血缘关系:与 GM100 task_00001 的关系(逐字节核验)

`lingbot-GM-100-demo/AgileX/BM-001` 与 GM100 项目的
`gm100-cobotmagic-lerobot-demo/task_00001` 是**同一份采集数据**的两次交付,
经逐字节对比,差异仅有:

| 部分 | 对比结果 |
|---|---|
| 130 个 data parquet | ✅ **字节级完全相同** |
| 390 个视频 | 384 个字节级相同;**6 个不同**(见 §2.2) |
| tasks.jsonl / episodes.jsonl / episodes_stats.jsonl | ✅ 完全相同 |
| meta/info.json | 仅 `robot_type` 不同:**`agilex_cobot_magic`**(本数据集) vs `songling_selfcollect`(GM100) |
| meta/quality_check.jsonl | 本数据集没有(GM100 那份也是 0 行空文件,无信息) |

因此 55D 映射与 FK 方案与 GM100 完全一致(§3)。

### 2.1 数据布局

每个压缩包解压后是一个 **lerobot v2.1** 数据集目录:

```
BM-001/
├── data/chunk-000/episode_XXXXXX.parquet    # 每集一个文件(428~1007 帧)
├── videos/chunk-000/<key>/episode_XXXXXX.mp4  # 每集每相机一个文件(~14-34s)
└── meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}
```

- task 只有一个:`hit-ball-with-gripper-to-score-goal`
- 帧率原生 30fps(无帧率归一问题);`action.*` 是命令目标,**不是**标准格式约定的"下一帧 state"

**源数据提供的通道**(episode parquet 的 12 列):

| 源列 | 宽度 | 说明 |
|---|---|---|
| `observation.state.arm.position` | 12 | 左臂6关节 + 右臂6关节(**弧度**) |
| `observation.state.arm.velocity` | 12 | 关节速度 |
| `observation.state.arm.effort` | 12 | 关节力矩 |
| `observation.state.effector.position` | 2 | 左/右夹爪开合(**原始量**,范围 [0, 0.08]) |
| `observation.state.effector.effort` | 2 | 夹爪力矩 |
| `action.arm.position` | 12 | **命令目标**(下发值,非实际到达值) |
| `action.effector.position` | 2 | 夹爪命令目标 |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | - | 索引列 |

关键诊断:无末端位姿(eef)数据;无子任务标注;无 URDF(已由 GM100 项目引入
Piper 臂 URDF 做 FK,见 §3.2)。

### 2.2 与 GM100 版本不同的 6 个视频(已逐一核对)

`camera_top` ep 27/51/63、`camera_wrist_left` ep 54/57/67 —— **帧数、时长、
分辨率、fps 与 GM100 版完全一致**(ffprobe `-count_frames` 逐对核验),
仅码率约为一半(重新编码过)。属交付侧的视频重编码,不改变内容帧数,
对转换与加载无影响;本转换**忠实保留本数据集版本的视频**(硬链接原文件)。

---

## 3. 55D 槽位映射(与 GM100 一致)

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
本机器人是 6-DOF,所以"**填 6 + 空 1**":左臂占 0:6、右臂占 7:13,第 6/13 维置 0
且 mask=0。这是规范"有什么填什么、缺的置空"的标准做法。

### 3.2 eef 槽:URDF 正运动学回填

源数据只有关节角、没有末端位姿。用 Piper 臂 URDF(`urdf/aloha_tracer2_dabai_dark.urdf`,
来源见 `urdf/README.md`)做**正运动学(FK)**,把关节角换算成末端位姿,回填 14:28
并置 mask=1。实现:`lingbot_fk.py`(与 GM100 项目同一模块,仅类名不同)。

- **eef 定义**:腕部法兰(URDF `link6` 帧)的位姿;映射 `left_arm_0..5 → fl_joint1..6`、
  `right_arm_0..5 → fr_joint1..6`
- **坐标系**:URDF 根帧 `footprint`(与臂安装面 `body_Link` 重合)= 机器人基座系;
  位置单位米,四元数 **xyzw**(55D 约定)
- 关节顺序/零位/符号的对齐验证(**URDF 限位 vs 数据分布对照、臂展核验**)
  已在 GM100 项目完成并记录(GM100 README §3.2);本数据集与 GM100 的数据逐字节相同,
  方案直接沿用
- **前提(待硬件确认)**:实体臂 = Piper 臂;若型号不符,需换 URDF 重算

**本次转换的独立核验**(校验脚本输出):

```
eef position range (m): left  x[0.141,0.743] y[0.088,0.385] z[0.890,1.282]
eef position range (m): right x[0.194,0.702]  y[-0.435,0.122] z[0.888,1.353]
```

左右臂 y 区间自然分离(左 y>0 / 右 y<0),z 高度 0.89~1.35m 合理,四元数模恒为 1。
**产物 data parquet 与 GM100 项目对同一数据的输出逐字节一致**(见 §6)。

### 3.3 action 语义:下一帧 state(不是命令目标)

规范 §3.2 约定 `action_unified[t] = state_unified[t+1]`(末帧复制自己,**严格在 episode 内 shift**)。
源数据的 `action.arm.position` 是**命令目标**,与实际到达值有实测差距
(GM100 项目实测最大 8.1°)。因此:**命令目标保留为透传列,action_unified 按 shift 生成**。

---

## 4. 输出结构

```
lingbot-GM-100-preprocess/
├── lingbot_extract.py    # ① 批量解压压缩包 -> 数据集目录(§4.0)
├── lingbot_convert.py    # ② 转换(§7)
├── lingbot_fk.py         # eef 正运动学模块(§3.2)
├── lingbot_validate.py   # ③ 校验(§7)
├── _progress.py          # 进度/ETA 报告器(标准库,无额外依赖)
├── urdf/                 # Piper 臂 URDF + 来源说明(§3.2)
├── README.md             # 本文档
└── processed_dataset/    # ← 所有转换产物
    └── BM-001/           # 每个数据集一个独立输出(保持源目录名)
        ├── data/chunk-000/file-000.parquet      # 16 列(见 §4.1),zstd,6.6 MB
        ├── videos/<key>/chunk-000/file-XXX.mp4  # 逐集硬链接(§4.3)
        ├── meta/
        │   ├── info.json                        # v3.0,fps=30,features+custom
        │   ├── tasks.parquet                    # 任务词表
        │   ├── subtasks.parquet                 # 空表(源无子任务标注)
        │   ├── episodes.jsonl                   # sidecar(action_config=[])
        │   ├── episodes/chunk-000/file-000.parquet  # episode 索引(定位 data/videos)
        │   └── modality.json                    # 55D 映射说明
        └── qc_report.json                       # L1 QC 摘要
```

### 4.0 解压(lingbot_extract.py)

- 递归扫描 `--src-root` 下的 `*.tar.gz` / `*.tgz`(默认 src-root 即数据集目录)
- 先解到隐藏临时目录 `.extracting_<包名>`,**校验每个顶层目录确实是 lerobot
  数据集**(meta/info.json + data/ + videos/)后再原子式移入正式位置 ——
  中断不会留下半成品
- **幂等**:已解压且结构完好的包直接跳过;重跑安全
- `--dest` 指定单独的解压根(默认解到压缩包旁边);`--force` 强制重解;
  `--continue-on-error` 单包失败不中断,末尾列出失败清单
- **绝不删除/修改源压缩包**;mp4 本身已压缩,解压后体积 ≈ 包体积,预留等量磁盘

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

按 lerobot 自身口径轮转(`data_files_size_in_mb=100`,比较**磁盘文件大小**)。
本数据集压缩比高(~12.7x),数据远小于 100MB,**只有一个 data 文件**(6.6 MB),未触发轮转。

### 4.3 视频:逐集硬链接(不合并、不重编码)

- 布局转换:`videos/chunk-000/<key>/episode_XXXXXX.mp4` → `videos/<key>/chunk-000/file-XXX.mp4`
  (`file_index == episode_index`);
- 用 `os.link` **硬链接**,零拷贝、零新增磁盘、零画质损失(已核验 inode 相同);
- **不做 500MB 合并**:标准格式 §5 明确"切分只在 episode 边界"、per-episode 小文件
  "索引列均已正确指向,加载无差别";本数据集每集 ~14-34s,远低于 960s 硬门槛;
- `info.json` 的 `video_files_size_in_mb` 登记为实际最大单文件大小(而非照抄 500)。

---

## 5. 已知数据特征(非缺陷,如实记录)

1. **与 GM100 的 6 个视频重编码差异**(§2.2):帧数/时长一致,仅码率不同;
   本转换保留本数据集版本,未与 GM100 版本合并。
2. **gripper 量纲与其他数据集不同**:本数据集夹爪值范围约 **[0, 0.08]**;
   agibot 是 ~[35, 120];egosuite 是 [0,1] 捏合度。转换阶段保留原始值,
   不强行归一化(归一化是下游训练职责)。
3. **无子任务标注**:`subtask_index` 恒 -1、`subtasks.parquet` 为空表(保留 schema)。
   若后续补充标注,可原地回填(逐帧列 + 词表 + episodes.jsonl 三件套)。
4. **eef 为 FK 推算值(非实测)**:由关节角 + Piper 臂 URDF 正解得到(§3.2),
   前提"实体臂 = Piper 臂"待硬件确认。
5. **robot_type 为 `agilex_cobot_magic`**(源 info.json 原样保留;GM100 交付中该字段
   为 `songling_selfcollect`,属同一数据的不同标注)。

---

## 6. 验证结果(全部通过)

转换后用 `lingbot_validate.py --full-videos` 全量校验,**ALL CHECKS PASSED (0 warnings)**:

| # | 检查 | 结果 |
|---|---|---|
| 1 | info.json: v3.0 / fps=30 / 全部 features 注册(含 `subtask_index` int64) | ✅ |
| 2 | data parquet 列序 16 列与规范一致;55D 列均为 fixed_size_list<float32>[55] | ✅ |
| 3 | mask/reliability 模式:active=[0..5,7..12,14..27,28,29],其余恒 0(逐集核验) | ✅ |
| 4 | 空槽(6、13、30:55)全为 0;eef 四元数模 ≈ 1(逐集核验) | ✅ |
| 5 | `action_unified[t] == state_unified[t+1]`,末帧复制自身(逐集核验) | ✅ |
| 6 | 帧数对账:Σlength == data 行数 == info.total_frames == 64,001 | ✅ |
| 7 | episodes 索引:`dataset_from/to_index` 连续覆盖 [0,N);视频定位列指向正确 | ✅ |
| 8 | 视频:每集每相机文件齐全;**全量 390 个**逐一核对时长 == length/30 | ✅ |
| 9 | 词表:tasks.parquet 1 行;subtasks.parquet 空表 schema 正确 | ✅ |
| 10 | **`LeRobotDataset` 官方加载成功**(lerobot 0.4.4,`len == 64001`) | ✅ |

**交叉验证(超出常规校验的额外核验)**:

- 产物 `data/chunk-000/file-000.parquet` 与 GM100 项目对同一数据的输出
  **逐字节一致**(证明 55D 构造/FK/透传全链路一致);
- 视频硬链接 inode 与源文件相同(零拷贝核验通过)。

速度参考:本数据集转换 ~8s,全量校验(含 390 视频)**~30s**。

---

## 7. 执行流程与批量指南

### 7.0 流程总览(压缩包 → 交付)

```
源:压缩包                         输出:每数据集一个 v3.0 数据集
  AgileX/BM-001.tar.gz              processed_dataset/BM-001/
        │  ① lingbot_extract.py 解压 + 结构校验(幂等/可续跑)
        ▼
      BM-001/(v2.1 布局)
        │  ② lingbot_convert.py 列映射 → 55D 构造 → eef FK 回填 → 透传列 →
        │                      ~100MB 轮转 → 视频硬链接 → meta/词表/episodes 索引
        ▼
      processed_dataset/BM-001/(v3.0 + 55D)
        │  ③ lingbot_validate.py 10 项全量校验(含 lerobot 官方加载软检查)
        ▼
      交付(ALL CHECKS PASSED 才算完成)
```

### 7.1 依赖与环境

| 类别 | 需要什么 | 说明 |
|---|---|---|
| Python | ≥ 3.10 | 脚本用到了 `list[dict]` 等新式类型标注 |
| **必需包** | `pyarrow` `numpy` `scipy` | 转换与 FK 的全部依赖(`extract` 只用标准库) |
| 可选包 | `av`(PyAV) | 校验时的视频时长核对(缺失只报 warning,不判失败) |
| 可选包 | `lerobot` | 校验时的官方加载(软检查,同上) |
| **不依赖** | ffmpeg | 视频是**硬链接**不做转码,转换全程不需要 ffmpeg |

本机环境(cosmos-framework,含全部可选包):
```bash
CONDA_HOME="/data5T/liujin/software/miniconda3"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate cosmos-framework
```

### 7.2 单数据集执行(以 BM-001 为例)

```bash
cd /data5T/liujin/code/blue_zone_code/data_preprocess/lingbot-GM-100-preprocess

# ① 解压(幂等,可反复执行)
python lingbot_extract.py --src-root /data5T/liujin/dataset/lingbot-GM-100-demo

# ② 转换(BM-001 已完成,重跑加 --skip-existing 即跳过)
python lingbot_convert.py --task BM-001 --workers 8
python lingbot_validate.py --task BM-001 --full-videos --workers 8   # 必须 ALL CHECKS PASSED 才算通过

# ③ FK 模块自检(秒级)
python lingbot_fk.py

# ④ 加载产物(可选)
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/lingbot_BM-001', root='processed_dataset/BM-001')
print(len(ds), ds.meta.tasks)
"
```

### 7.3 大批量完整示例(假设有 N 个压缩包)

```bash
# ── 第 0 步:环境准备(一次性) ──────────────────────────────────
conda activate cosmos-framework
python -c "import pyarrow, numpy, scipy; print('deps ok')"
python lingbot_fk.py                         # FK 自检

# ── 第 1 步:解压全部压缩包(幂等;重复执行只解新包) ──────────────
python lingbot_extract.py --src-root /data5T/liujin/dataset/lingbot-GM-100-demo \
    --workers 8 2>&1 | tee extract.log
#   期望:每包一行 [OK]/[SKIP] + [PROGRESS];EXTRACT_DONE ok=.. skip=.. failed=0
#   有 FAIL 时:看日志定位,修好后原样重跑(已成功的自动 SKIP)

# ── 第 2 步:pilot 单数据集转换 + 校验 ──────────────────────────
python lingbot_convert.py --task BM-001 --workers 8
python lingbot_validate.py --task BM-001 --full-videos --workers 8
#   必须通过后再进下一步

# ── 第 3 步:全量转换(后台 + 日志;可断点续跑) ─────────────────
setsid nohup python lingbot_convert.py --workers 8 --skip-existing --continue-on-error \
    > convert.log 2>&1 &
tail -f convert.log                        # [OK]/[SKIP]/[FAIL] + [PROGRESS] 行
#   若中断/崩溃:原样重跑同一条命令 —— 已完成的显示 [SKIP],自动接着转

# ── 第 4 步:全量校验(转换完成后) ───────────────────────────────
python lingbot_validate.py --all --workers 8 > validate.log 2>&1
tail -1 validate.log
#   期望:RESULT: ALL CHECKS PASSED (0 warnings, N tasks)
#   若失败:日志末尾有 FAILED_TASKS: xxx;重转这些数据集后再校验

# ── 第 5 步:核对结果 ───────────────────────────────────────────
grep -c "^\[OK\]" convert.log              # 成功数(应=[OK]+[SKIP] 总数)
grep "FAILED_TASKS" convert.log            # 失败清单(应无输出)
ls processed_dataset | wc -l               # 数据集目录数(应=N)
```

**时间/空间参考**(按 BM-001 规模:314 MB 包 / 13 万帧级):
- 解压:约 3 秒/包(HDD 可能更慢);mp4 已压缩,解压后体积 ≈ 包体积 → 预留等量磁盘
- 转换:约 8 秒/数据集 → N 个近线性
- 校验:带 `--full-videos` 约 30 秒/数据集(390 个视频逐一核对)
- 视频默认硬链接(**零占用**;源与输出须同一文件系统,跨文件系统自动退化拷贝,
  见 §7.4)

### 7.4 批量注意事项(重要)

- **同文件系统要求**:视频默认 `os.link` 硬链接,要求源(解压后的数据集)与输出
  在同一文件系统。跨文件系统时脚本**自动退化为拷贝**(日志 WARN;
  `info.json.custom.videos.n_copied` 记录),此时输出会真实占用与源视频等量磁盘。
  本机 `/data5T/liujin/dataset` 与 `/data5T/liujin/code` 同为 `/dev/sdb1`,满足硬链接条件。
- **幂等与中断恢复**:解压、转换对每个数据集先重建再写入,完成后落标记
  (转换写 `.convert_done`);**大批量务必带 `--skip-existing` 重跑**。
  校验是只读的,可反复执行。
- **磁盘 IO 是瓶颈**:批量以串行为主,并行度不宜过高;长任务用 `setsid nohup`
  后台运行并留日志(参考 `standard_format/docs/EGOSUITE_PIPELINE_RUNBOOK.md`)。
- **完成判据**:每批转换后必须跑校验,输出 `ALL CHECKS PASSED` 才算完成。

### 7.4.1 进度与并行

三个脚本默认每 10 秒输出一行进度(URL: `--progress-interval N`,关闭: `--no-progress`):

```
[PROGRESS] extract 23/300 archives (7.7%) elapsed 00:01:12 ETA 00:14:26 0.3 archives/s | BM-023.tar.gz
[PROGRESS] convert 1,234,567/10,000,000 frames (12.3%) elapsed 00:32:10 ETA 03:49:00 | BM-042
[PROGRESS] validate 8,900,000/10,000,000 frames (89.0%) elapsed 00:12:05 ETA 00:01:30 | BM-291
```

- `--workers N`(默认 8):extract 并行解压 N 个压缩包、convert 并行转换 N 个数据集、
  validate 并行校验 N 个数据集;`1` 表示串行。
- `validate --video-workers N`(默认 4):`--full-videos` 时并行探测 N 个视频时长。
- **输出完全确定**:每个数据集输出目录独立,单数据集内仍串行;并行不改变列、索引、
  视频命名或文件划分。GM100/lingbot 的 demo 产物已与改造前逐字节对比一致。
- 解压的最终发布步骤加了跨进程文件锁(`.extract_publish.lock`),多个压缩包解到同一
  `--dest` 时不会因同名顶层目录互相踩踏。
- 并行度建议:默认 8;HDD/单卷场景先用 `iostat -x 5` 观察磁盘,再按实测调整。
  每个 convert worker 已把 pyarrow CPU 线程数限制为 1。

### 7.5 迁移到新压缩包/新数据集时改什么

数据集无关的机制(55D 构造 / mask / 轮转 / meta 与词表 / episodes 索引 / 视频链接 /
解压 / 批量恢复)可直接复用。若新的压缩包仍是**同一机器人**(同列名)的数据:
**无需改任何代码** —— 解压 → `lingbot_convert.py`(递归自动发现新数据集)→ 校验即可。

若换了数据源格式(列名/维度不同),要改的地方:

| 要改的地方 | 位置 | 说明 |
|---|---|---|
| 源列名 | `lingbot_convert.py` 顶部 `SRC_ARM_POS` / `SRC_EFF_POS` / `PASSTHROUGH_COLS` | 新数据集的动作/状态列可能改名 |
| 55D 槽位映射 | `build_episode_table()` 及 `LEFT_ARM_JOINT` 等常量 | 例:新数据直接有 eef 而无关节角,则改填 eef 槽 |
| FK(仅"关节角→eef"需要) | `--urdf` 指向新臂 URDF;`lingbot_fk.py` 中 `fl`/`fr` 前缀按新 URDF 命名调整 | 不同臂型/命名规则 |
| 相机与分辨率 | **无需改**:自动从 `videos/chunk-000/` 与源 `info.json` 探测 | |
| 数据集目录命名 | **无需改**:递归发现任何含 `meta/info.json` 的目录 | 重名冲突会报错提示 |

### 7.6 迁移到新环境(一个文件夹下几百个压缩包)

**迁移清单**(拷到新环境的最小文件集,保持相对位置不变):

```
lingbot-GM-100-preprocess/
├── lingbot_extract.py     # ① 批量解压
├── lingbot_convert.py     # ② 转换
├── lingbot_fk.py          #    (convert 依赖)
├── lingbot_validate.py    # ③ 校验
├── _progress.py           #    进度/ETA(三个脚本依赖)
└── urdf/aloha_tracer2_dabai_dark.urdf   # (convert 依赖)
```

- 不依赖 cwd、不依赖本机路径(convert 通过**脚本所在目录**解析 fk 模块与 urdf/)
- 依赖仅 `pyarrow` `numpy` `scipy`(extract 只用标准库);可选 `av`/`lerobot`
- **新旧环境路径不同时,显式传 `--src-root` / `--dest` / `--out`**(脚本内默认值仅本机有效)

**场景:新环境 `/data/newenv/archives/` 下有几百个 `BM-*.tar.gz`**:

```bash
# ── 0. 一次性准备 ──
conda activate cosmos-framework            # 或 pip install pyarrow numpy scipy
python lingbot_fk.py                       # FK 自检(秒级,验证 URDF 可解析)

# ── 1. 解压全部压缩包(幂等;中断后原样重跑,已解压的自动 SKIP) ──
python lingbot_extract.py --src-root /data/newenv/archives --dest /data/newenv/datasets \
    --workers 8
#   期望 EXTRACT_DONE ok=N skip=0 failed=0;有 FAIL 看日志修好后原样重跑

# ── 2. pilot:先试一个数据集,走通再全量 ──
python lingbot_convert.py  --src-root /data/newenv/datasets --out /data/newenv/processed \
    --task BM-001 --workers 8
python lingbot_validate.py --out /data/newenv/processed --task BM-001 --full-videos --workers 8
#   必须 ALL CHECKS PASSED 再进下一步

# ── 3. 全量转换(后台;中断/崩溃后原样重跑同一条命令即可续) ──
setsid nohup python lingbot_convert.py --src-root /data/newenv/datasets \
    --out /data/newenv/processed --workers 8 --skip-existing --continue-on-error \
    > convert.log 2>&1 &
tail -f convert.log                         # [OK]/[SKIP]/[FAIL] + [PROGRESS] 行

# ── 4. 全量校验 ──
python lingbot_validate.py --out /data/newenv/processed --all --workers 8 > validate.log 2>&1
tail -1 validate.log                        # 期望 RESULT: ALL CHECKS PASSED
```

**磁盘预估**(按单包 314 MB / 13 万帧级):解压后 ≈ 包体积(mp4 已压缩),
即几百个包需预留"包总体积"等量的解压空间;转换产物 data ≈ 7 MB/数据集,
视频在**源与输出同一文件系统时硬链接零占用**(跨文件系统自动退化为真实拷贝,
按视频总量再预留)。粗略按 **"包总体积 ×2"** 备盘。

**失败恢复**:转换/校验都会在末尾打印 `FAILED_TASKS: ...`;修复原因后
**重跑同一条命令**(转换带 `--skip-existing`),已完成的自动 [SKIP],只重做失败的。

**批量实测(2026-10-15)**:一个文件夹放 2 个压缩包模拟该场景,全流程(解压 →
`--skip-existing` 转换 → `--all` 校验)跑通并**两次幂等重跑**验证
(解压/转换重跑全部正确 SKIP),`RESULT: ALL CHECKS PASSED (0 warnings, 2 tasks)`。

---

## 8. 决策记录(为什么这么做)

| 决策 | 结论 | 依据 |
|---|---|---|
| 解压流程内置 | 独立脚本 `lingbot_extract.py`,幂等 + 临时目录原子发布 + 结构校验 | 大批量输入为若干压缩包;中断/重跑安全(用户要求) |
| 55D 映射 | 与 GM100 完全一致(joint+gripper+**FK eef**) | 数据与 GM100 task_00001 逐字节相同;FK 模块已在对齐验证过的前提下复用 |
| action 语义 | `action[t]=state[t+1]`,命令目标进透传 | 规范 §3.2;命令目标与实际差最大 8.1°(GM100 实测),不可混用 |
| eef 来源 | FK 回填(mask=1) | Piper 臂 URDF;前提"实体臂=Piper"待硬件确认(与 GM100 相同) |
| 关节槽 | 6-DOF 填 6 空 1 | 槽宽 7 是全数据集上限(agibot 7-DOF);"有什么填什么" |
| gripper | 原始值,不归一化 | 与 agibot 一致;跨数据集归一化是下游职责 |
| 子任务 | 恒 -1,空词表 | 源无标注;保留 schema 便于未来回填 |
| 视频 | 逐集硬链接,不合并;保留本数据集版本(6 个重编码视频不替换) | 规范允许 per-episode 文件;零拷贝;如实交付源包内容 |
| robot_type | 原样保留 `agilex_cobot_magic` | 源 info.json 为准 |
| 轮转口径 | 磁盘大小(lerobot 同口径),自校准压缩比 | lerobot `data_files_size_in_mb` 按磁盘文件大小比较 |
