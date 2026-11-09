# robocoin_v4_run 使用说明

> 本文档基于对包内 `RUNBOOK.txt`、`run.sh`、`config.sh`、`scripts/convert_repos.py`、
> `scripts/verify.py`、`final_acceptance.py` 及 `vendor/standard_format/tools/` 的通读整理，
> 用于记录**改哪里、处理逻辑、产物结构**。权威口径仍以 `RUNBOOK.txt` 和
> `robocoin_converter_v4/README.md` 为准。

---

## 0. 这个包是做什么的

把**已下载、已解压**的 RoboCOIN 数据（LeRobot v2.1 布局）转换成团队统一格式：

- LeRobot **v3.0** 布局；
- 55D 统一 `state_unified` / `action_unified`，配套 `mask_state` / `mask_action`；
- 原始列全部保留为 `source.*`；
- 原始 meta / annotations / README 另存到输出；
- 按「仓库 × split」拆成独立数据集，不跨机器人混拼。

它**不做**：下载、解压、修改输入、训练、策略评估。

### 包内组成

| 路径 | 作用 |
|---|---|
| `RUNBOOK.txt` | 官方操作手册（最短路径） |
| `config.sh` | 唯一需要修改的配置文件 |
| `run.sh` | 命令入口：`plan/convert/verify/load/accept/status/all` |
| `final_acceptance.py` | 全目录验收（要求 809 仓库完整覆盖） |
| `robocoin_converter_v4/` | 转换器代码、vendored 标准工具、元数据与历史报告 |
| `robocoin_loader_tests_v4/` | LeRobot 0.5.0 实际加载测试 |
| `PACKAGE_SHA256SUMS` | 打包校验清单（完整包校验用） |

**包内不含**：输入数据、Python 虚拟环境、模型/训练代码。

---

## 1. 你需要修改哪里

### 1.1 `config.sh`（唯一必改文件）

```bash
# 转换环境：Python 3.10 + robocoin_converter_v4/requirements.txt
CONVERT_PY=./.venv-convert/bin/python
# 加载环境：LeRobot 0.5.0
LEROBOT_PY=./.venv-lerobot/bin/python

INPUT=./input       # 已解压数据目录
OUTPUT=./output     # 转换产物
WORK=./work         # 计划/日志/报告/断点信息

WORKERS=4           # 并发仓库/split 进程数
VIDEO_WORKERS=4     # 每个进程内并发的视频任务数
DECODE_THREADS=8    # 每个视频任务的解码线程数
CPU_BUDGET=128      # CPU 预算（会设置进程 affinity）
VIDEO_MB=128        # 输出单个视频文件的目标上限（十进制 MB）
```

| 变量 | 说明 | 注意 |
|---|---|---|
| `CONVERT_PY` | 转换解释器 | 需装 `numpy==1.26.4`、`pyarrow==25.0.1`、`pandas==2.3.3`、`av==17.1.0`、`jsonschema==4.26.0`；**不需要** PyTorch/LeRobot/GPU |
| `LEROBOT_PY` | 加载测试解释器 | 需 **LeRobot 0.5.0**（及 torch 等依赖） |
| `INPUT` | 源数据根目录 | 处理期间**必须保持不变**；可绝对路径或软链 |
| `OUTPUT` | 输出根目录 | 必须与 `INPUT`、`WORK` **互不包含**；v4 不要混入 v3 旧结果 |
| `WORK` | 工作目录 | 同上；保存 `plan.json`、`progress.json`、日志和报告 |
| `WORKERS` / `VIDEO_WORKERS` / `DECODE_THREADS` | 三级并行度 | 三者乘积必须 ≤ `CPU_BUDGET` |
| `CPU_BUDGET` | CPU 上限 | 脚本会把进程 affinity 限制到前 N 个 CPU |
| `VIDEO_MB` | 视频打包上限 | 默认 128 → 单个输出视频目标 ≤ 128,000,000 字节；单个超大 episode 不强制切分 |

**并行度建议**

- 128 CPU：默认 `4×4×8=128` 即可；
- 64 CPU：改成 `4×4×4`，并设 `CPU_BUDGET=64`；
- 实际可用的 affinity 小于 `CPU_BUDGET` 时，脚本会按实际可用核数校验，参数需要相应调小。

### 1.2 准备两个 Python 环境

```bash
# 转换环境
python3.10 -m venv .venv-convert
./.venv-convert/bin/python -m pip install -r robocoin_converter_v4/requirements.txt

# 加载环境（示例；lerobot 版本必须是 0.5.0）
python3.10 -m venv .venv-lerobot
./.venv-lerobot/bin/python -m pip install "lerobot==0.5.0"
```

> `config.sh` 里的路径也可以写绝对路径，或直接写命令名（如 `python3.10`）。

### 1.3 准备 INPUT

推荐先用软链指向数据盘：

```bash
ln -s /path/to/robocoin_extracted input
```

要求见下一节。

---

## 2. 输入数据要求

### 2.1 目录布局

```
INPUT/
├── <发布仓库名 A>/                # 保留 HF 或 ModelScope 的仓库名
│   ├── meta/
│   │   ├── info.json             # codebase_version 必须为 v2.1
│   │   ├── episodes.jsonl        # episode_index / length
│   │   └── tasks.jsonl           # 必需
│   ├── data/
│   │   └── chunk-000/episode_000000.parquet
│   ├── videos/
│   │   └── chunk-000/<video_key>/episode_000000.mp4
│   └── annotations/              # 可选：subtask_annotations.jsonl
└── <发布仓库名 B>/
    └── ...
```

### 2.2 识别规则

- `convert_repos.py` 用 `os.walk(INPUT)` 找「直接包含 `meta`、`data`、`videos` 任一子目录」的目录作为数据集根目录，找到后不再向下遍历。
- 目录名必须能在 `research/repository_aliases.json` 中匹配到 HF 或 ModelScope 仓库名；不认识的名字会被跳过并报：
  `Unknown repository directory name`。
- 同一仓库不能出现两份（HF 名 + ModelScope 名、或两份本地副本）；出现重复时**所有副本都会被跳过**。
- 支持多层目录，只要数据集根目录名正确即可。
- `info.json` 的 `data_path` / `video_path` 是路径模板，实际文件按 episode 编号格式化后必须存在且非空。

### 2.3 预检硬性条件

每个候选仓库必须全部满足：

1. `meta/info.json` 存在，且 `codebase_version == "v2.1"`；
2. schema 在 `research/validated_schemas.json` 白名单中，或能匹配 `research/dimension_repairs.json` 的修复规则；
3. `data/`、`videos/`、`meta/tasks.jsonl` 存在；
4. `meta/episodes.jsonl` 的 episode 数、帧数与 `info.json` 的 `total_episodes` / `total_frames` 一致；
5. episode ID 连续无缺口（`0..N-1`）；
6. 每个 episode 引用的 parquet 和所有相机视频文件都存在且非空；
7. split 范围合法、不重叠；未被 split 覆盖的 episode 会生成 `unassigned_<ep>` 单元；
8. 需要名称修复的仓库，`validate_vectors()` 会扫描**全部本地 parquet 行**，确认每行向量宽度与声明一致后才允许修复。

不满足的仓库不会中断整体流程，会记录到 `plan.json.skipped`。

---

## 3. 命令用法

所有命令都在 `robocoin_v4_run/` 下执行（`run.sh` 会自动 `cd` 到自己所在目录）：

```bash
cd /mnt/sfs_turbo/liujin/code/blue_zone_code/data_preprocess/robocoin_v4_run
bash run.sh <stage>
```

| 命令 | 作用 | 产物 |
|---|---|---|
| `bash run.sh plan` | 只做预检与计划，不转换 | `work/plan.json`，并打印任务数、跳过项、缺失仓库、磁盘预算 |
| `bash run.sh convert` | 生成计划并执行转换 | `OUTPUT/<unit>/...`、`work/progress.json`、`work/<unit>/profile.json`、`manifest.json` |
| `bash run.sh verify` | 逐 episode 保真校验 | `work/verification.json` |
| `bash run.sh load` | 用 LeRobot 0.5.0 实际加载 | `work/official_load.json` |
| `bash run.sh accept` | 全目录验收汇总 | `work/acceptance_v4.json` |
| `bash run.sh status` | 查看计划与进度 | 终端 JSON |
| `bash run.sh all` | 依次执行 `convert → verify → load → accept` | 上述全部 |

**注意：**

- `all` **不包含** `plan`；但 `convert` 内部会重新扫描输入并生成 `work/plan.json`。
- `run.sh` 使用 `set -euo pipefail`：任一阶段失败会停止后续阶段。转换阶段即使个别 job 失败，`convert_repos.py` 也会以退出码 1 结束；修好后重跑 `bash run.sh all` 即可续跑。
- 长任务建议后台运行：

```bash
nohup bash run.sh all > launcher.log 2>&1 < /dev/null &
bash run.sh status
tail -f launcher.log
```

- 每个阶段的完整日志都会写到 `work/logs/<stage>_<时间>_<pid>.log`。
- 配置文件路径可用环境变量覆盖：`CONFIG=/path/to/config.sh bash run.sh plan`。

### 3.1 直接调用 Python 脚本的场景

`run.sh` 没有暴露 `--only-repairable`、`benchmark`、`audit_numeric_semantics`，需要时直接调用：

```bash
# 只处理 54 个缺名修复仓库（写到新的 OUTPUT/WORK）
./.venv-convert/bin/python robocoin_converter_v4/scripts/convert_repos.py convert \
  --input "$INPUT" --output /data/robocoin_recovered54_v4 --work /data/robocoin_recovered54_work \
  --only-repairable \
  --workers 4 --video-workers 4 --decode-threads 8 --cpu-budget 128

# 数值语义审计（只读 Parquet，不改数据）
./.venv-convert/bin/python robocoin_converter_v4/scripts/audit_numeric_semantics.py \
  --input "$INPUT" --report "$WORK/semantic_audit.json" --workers 8

# 性能基准（必须使用全新目录）
./.venv-convert/bin/python robocoin_converter_v4/scripts/benchmark.py \
  --input /data/representative_subset --output /data/bench_output --work /data/bench_work \
  --report /data/bench_report.json --configs 16x4x2,8x4x4,4x4x8
```

> 直接调用 `convert_repos.py` 时，未显式传的参数使用脚本默认值：
> `--workers 16`、`--video-workers 4`、`--decode-threads 1`、`--cpu-budget 128`、`--video-mb 128`。

---

## 4. 处理逻辑

整体链路：

```
扫描 INPUT
  → 识别仓库（目录名 → canonical repo）
  → 预检（格式/结构/episode/文件/split）
  → 生成 plan.json
  → 每个 (仓库, split) 一个 job
      → profile.py：按字段名生成 55D 映射
      → from_lerobot.py：解析源 parquet / 视频，生成 manifest
      → enrich_annotations.py：拼接 subtask 语言
      → fast_media.py + media.py：视频无损合并
      → convert.py：逐帧映射 55D、写 LeRobot v3、QC
      → 复制原始 meta / annotations
      → 生成 conversion_receipt.json（签名 + 全文件 SHA256）
      → 原子发布
  → verify.py：逐 episode 源-输出比对、55D 独立复核、视频 packet 比对、RGB 读取
  → official_load.py：LeRobot 0.5.0 实际加载
  → final_acceptance.py：全目录验收
```

### 4.1 plan：预检与任务拆分

`scripts/convert_repos.py` 的 `plan` 阶段会：

1. 解析 `--input/--output/--work` 为绝对路径，并断言三者互不包含；
2. 创建 `OUTPUT` / `WORK`，并对 `OUTPUT/.lock` 加排他锁（防止并发跑同一输出）；
3. 扫描候选数据集根目录；
4. `repository_catalog.py` 通过 `research/repository_aliases.json` 把目录名解析为 canonical repo；
5. 检查重复仓库、源格式、schema、必需路径、episode 一致性、每个 episode 的文件存在性、split 合法性；
6. 对需要修复的仓库执行全量 parquet 向量宽度扫描；
7. 生成 jobs（每个 job 对应一个仓库的一个 split），并计算：
   - `input_bytes`：该仓库输入文件总字节；
   - 磁盘估算：`2 × input_bytes + 4096 × total_frames + 2 GiB`；
8. 写出 `work/plan.json`。

`plan` 还会额外打印：

```json
{
  "jobs": ...,
  "skipped": [...],
  "missing_repositories": [...],
  "input_TiB": ...,
  "additional_required_TiB": ...,
  "free_TiB": ...,
  "space_check_passed": true/false
}
```

### 4.2 convert：逐 job 转换

每个 job：

1. 读源 `meta/info.json`，用 `profile.py` 生成 55D 映射配置；
2. 计算签名 `signature`：
   - 源指纹（文件相对路径、大小、mtime；JSON/JSONL/YAML 还包含内容哈希）
   - 映射配置
   - 代码哈希（`scripts/*.py`、`vendor/.../tools/*.py`、`research/dimension_repairs.json`、`research/repository_aliases.json`）
   - `decode_threads`
3. **断点判断**：如果目标目录已存在，读 `conversion_receipt.json`：
   - 签名一致且 inventory 中每个文件哈希一致 → `skipped_verified`；
   - 否则直接报错：`Existing output differs; choose new output version`（fail closed，不覆盖旧结果）；
4. 在 `OUTPUT/.<unit>.pending/` 中构建：
   - `from_lerobot.adapt()`：生成 episode manifest（源 parquet 引用、相机视频路径、任务词表、split）；
   - `enrich_annotations.enrich()`：按发布顺序拼接 subtask 标签，生成 `segments`；
   - `fast_media.install()` + `convert.py`：
     - 视频：**只做无损 packet copy / 拼接，不重编码**；同相机必须格式一致；按 `VIDEO_MB` 目标大小分组；
     - 数据：逐帧构建 55D、mask、`source.*` 原始列、时间列、语言列、动作语义列；
     - 写 `meta/info.json`、`stats.json`、`tasks.parquet`、`subtasks.parquet`、`episodes.jsonl`、
       `episodes/chunk-000/file-000.parquet`、`video_timing/episodes.parquet`、
       `input_manifest.json`、`media_manifest.json`、`qc/decode.json`、`qc_report.json`、`SUCCESS.json`；
5. 复制原始 `meta/` → `meta/source_metadata/`；有 `annotations/` → `meta/source_annotations/`；
   README / info.yaml 若存在也复制；
6. 再次校验源指纹未变化；
7. 生成 `conversion_receipt.json`：`signature`、`inventory`（输出每个文件的路径、字节数、SHA256）、
   `source`、`repository`；
8. 原子重命名 `.<unit>.pending` → `OUTPUT/<unit>/`；
9. 更新 `work/progress.json`（每个 job 完成/失败都会写一次）。

### 4.3 55D 映射规则（profile.py）

55D 槽位（从 0 开始）：

| 槽位 | 内容 |
|---|---|
| 0:7 | 左臂关节（rad） |
| 7:14 | 右臂关节（rad） |
| 14:21 | 左 EEF（xyz + xyzw） |
| 21:28 | 右 EEF（xyz + xyzw） |
| 28 / 29 | 左 / 右夹爪 |
| 30:36 | 左手手指 |
| 36:42 | 右手手指 |
| 42:46 | 腰部 |
| 46:48 | 头/颈 |
| 48:51 | 底盘 |
| 51:55 | 保留 |

关键策略：

- **关节按名称匹配**：左右臂各必须 6 或 7 个关节，否则该仓库失败；手部超过 6 维时完整保留在 `source.*`，不截断；
- **夹爪**：保留原始数值尺度（可能是 0–100 / 0–1000 / 角度），越大越开，不裁剪、不归一化；
- **末端位姿**：优先使用原生 SDK 的 `xyz + xyzw`（约 30 个仓库）；否则只用仿真列的 xyz，不猜 Euler 约定；四元数会归一化并屏蔽孤立 >120° 跳变；
- **缺失/未映射的槽位**：值置 0，对应 mask=0；
- **原始列**：`passthrough_columns='all'`，全部进入 `source.*`。

### 4.4 动作语义（episode_policy.py）

对每个 episode 独立判定：

- 默认 `action_source_kind=0`：保留发布 action；
- 若整集所有已映射字段和共享具名原始分量**逐值完全相同**、mask 一致、且整集不是静止，则判定为复制关系，改成下一帧 state 派生（`action_source_kind=1`），末帧 action 全 0、mask 全 0；
- 少于 3 帧、存在非有限值、字段语义不一致等情况一律保留发布 action；
- 决策写入 `meta/episodes.jsonl` 的 `action_semantics`，同时输出 `action_source_kind` / `action_target_step` 列。

### 4.5 verify：保真校验

`scripts/verify.py` 对 `plan.json` 中每个 job：

1. 校验 `SUCCESS.json` 和 `conversion_receipt.json`，重算 inventory 中每个输出文件的 SHA256；
2. 读输出 `meta/input_manifest.json` 和源 `meta/info.json`，按 episode 重读源 parquet；
3. 对比源每一列与输出 `source.*`（逐行、逐值）；
4. 用独立实现的映射逻辑重算 55D，逐值比对 `state_unified/action_unified` 和 mask；
5. 检查 `sample_dt`、时间戳、动作语义列；
6. 对每个输出视频 demux 所有 packet，比对 packet 大小/内容/顺序哈希；
7. 通过 `StandardDataset` 实际读取首/中/末帧 RGB；
8. 写 `work/verification.json`。

`verification.json` 的 `passed=true` **只代表计划内输出通过**；还要看 `corpus_complete`、`skipped`、
`missing_expected_repos`。子集运行出现 `corpus_complete=false` 是正常的。

### 4.6 load：LeRobot 0.5.0 实际加载

`robocoin_loader_tests_v4/scripts/official_load.py`：

1. 先断言 `LEROBOT_PY` 的 LeRobot 版本为 `0.5.0`；
2. 对 `OUTPUT` 下每个含 `SUCCESS.json` 的单元，用子进程运行 `loader_probe.py`（超时 300 秒/单元）；
3. `loader_probe.py` 使用包内 `lerobot_compat.Unified55LeRobotDataset`（薄兼容层，处理
   `subtask=-1` 和 singleton list 列），检查：
   - `state_unified/action_unified/mask_state/mask_action` 形状与数值；
   - 每集边界（首帧、末帧、前后帧）加载；
   - 图像 RGB 与 `StandardDataset` 逐像素一致；
4. 写 `work/official_load.json`。

### 4.7 accept：全目录验收

`final_acceptance.py` 是 fail-closed 的严格验收，要求同时满足：

- `plan.selection == "all"`，`expected_training_repos == 809`；
- 无 `skipped`、无 `missing_expected_repos`、无重复仓库；
- 计划 job 集合、`progress.json`、`verification.json`、`official_load.json` 的单元集合完全一致；
- 所有转换 job 状态为 `complete` 或 `skipped_verified`；
- `verification.passed == true` 且 `corpus_complete == true`；
- `official_load.passed == true` 且 `lerobot_version == "0.5.0"`；
- `OUTPUT` 下的实际目录集合与计划一致；
- 每个单元有 `SUCCESS.json` / `conversion_receipt.json`，`meta/info.json` 为 v3.0，
  帧数/集数与校验报告一致；
- 最后写 `work/acceptance_v4.json`，只有全部通过才 `passed=true`。

> **重要**：`accept` 是按 809 仓库全量设计的。用小子集试跑时，`accept` 必然失败；
> 这属于预期行为，不代表转换本身有问题。

### 4.8 status

读取 `work/plan.json` 和 `work/progress.json`，输出：

```json
{
  "planned_jobs": ...,
  "reported_jobs": ...,
  "status_counts": {"complete": ..., "skipped_verified": ..., "failed": ...},
  "plan_skipped": ...,
  "missing_repositories": ...,
  "failed": [...]
}
```

注意：`progress.json` 每次运行都会被重写，不是跨运行的历史累计。

---

## 5. 产物结构

### 5.1 OUTPUT（每个 job 一个独立数据集目录）

目录名：

```
<canonical_repo_name>__<输入相对路径的 sha256 前 8 位>__<split>
```

例如：`Agilex_Cobot_Magic_wipe_table__a1b2c3d4__train/`

```
OUTPUT/<unit>/
├── SUCCESS.json                       # 成功标记（仅全部完成后写入）
├── conversion_receipt.json            # 签名 + 全文件 SHA256 inventory
├── qc_report.json                     # 输出结构/数值 QC 报告
├── data/
│   └── chunk-000/file-000.parquet     # 每文件约 5 万行
├── videos/
│   └── observation.images.<cam>/
│       └── chunk-000/file-000.mp4     # 无损 packet copy/拼接
└── meta/
    ├── info.json                      # LeRobot v3.0 元数据 + custom 段
    ├── stats.json                     # 55D 等统计（按 mask）
    ├── tasks.parquet / subtasks.parquet
    ├── episodes.jsonl                 # 逐集扩展元数据（action_semantics 等）
    ├── episodes/chunk-000/file-000.parquet
    ├── video_timing/episodes.parquet
    ├── input_manifest.json            # 完整映射/来源说明（= verify 的依据）
    ├── media_manifest.json            # 视频打包与源文件映射
    ├── qc/decode.json
    ├── source_metadata/               # 原始 meta（未修改）
    ├── source_annotations/            # 原始 annotations（若存在）
    ├── metadata_repair.json           # 仅 54 个缺名仓库有
    └── source_info_effective.json     # 仅 54 个缺名仓库有
```

### 5.2 WORK

```
WORK/
├── plan.json                 # 预检结果：jobs/skipped/missing/磁盘估算
├── progress.json             # 本次运行的逐 job 结果（每次重写）
├── logs/
│   └── <stage>_<日期>_<pid>.log
├── <unit>/profile.json       # 该单元的 55D 映射配置
├── <unit>/manifest.json      # 源→输出 episode/视频清单
├── verification.json         # verify 报告
├── official_load.json        # LeRobot 加载报告
└── acceptance_v4.json        # 最终验收报告
```

### 5.3 关键报告字段

| 文件 | 关键字段 | 含义 |
|---|---|---|
| `plan.json` | `valid_jobs`、`skipped`、`missing_expected_repos`、`duplicate_repositories`、`extra_disk_estimate_bytes` | 计划与预检结果 |
| `progress.json` | `results[].status`：`complete` / `skipped_verified` / `failed` | 本次运行结果 |
| `verification.json` | `passed`、`corpus_complete`、`units[].rows`、`units[].video_packets` | 保真校验 |
| `official_load.json` | `passed`、`lerobot_version`、`units[].passed` | 加载测试 |
| `acceptance_v4.json` | `passed`、`errors`、`repositories`、`output_units`、`episodes`、`frames` | 全量验收 |

---

## 6. 断点续跑与失败处理

- **续跑**：保持代码、`config.sh`、`INPUT` 不变，重新执行 `bash run.sh all`。
  - 已完成的单元：签名一致 + inventory 哈希一致 → `skipped_verified`；
  - 未完成单元：在隐藏 pending 目录重做；
  - 恢复粒度是「仓库/split」，不是视频内部。
- **不要修改运行中的代码或输入**；代码/输入/解码参数变化会使签名变化，脚本会拒绝复用旧结果。
- **失败会保留 `OUTPUT/.<unit>.pending/`**，用于诊断；不要盲目删除。
- **v4 必须使用新的 OUTPUT/WORK**；不要把 v3 结果混入，也不要拿 v3 结果当 v4 断点。
- `--only-repairable` 只处理 54 个缺名仓库，且应写到新的 OUTPUT/WORK。

---

## 7. 常见错误与处理

| 现象 / 报错 | 原因 | 处理 |
|---|---|---|
| `Unknown repository directory name: ...` | 目录名不是发布仓库名 | 改成 HF/ModelScope 仓库名（见 `research/repository_aliases.json`） |
| `Duplicate HF/ModelScope or local copies ...` | 同一仓库存在两份 | 只保留一份完整版本 |
| `Unverified source format` | `codebase_version != v2.1` | 确认输入是 v2.1 布局 |
| `Missing required source paths` | 缺 `data/`、`videos/` 或 `meta/tasks.jsonl` | 补齐源数据 |
| `Metadata count mismatch` | episodes.jsonl 与 info.json 的集数/帧数不一致 | 核对源数据 |
| `Gapped episode ids require explicit review` | episode 编号不连续 | 需要人工确认 |
| `Missing/empty referenced file` | `info.json` 模板格式化后找不到文件 | 核对路径模板与文件 |
| `Numeric schema not covered ...` | schema 不在白名单且无修复规则 | 记入 skipped，需人工审计 |
| `Missing/ambiguous component names` | state/action 的 names 缺失/重复 | 仅 54 个已审计仓库有修复规则 |
| `Existing output differs; choose new output version` | 代码/输入/参数变化导致签名变化 | 使用新的 OUTPUT/WORK |
| `Requested parallelism exceeds CPU budget` | 三级并行度乘积 > affinity | 调小 `WORKERS`/`VIDEO_WORKERS`/`DECODE_THREADS` 或增大 `CPU_BUDGET` |
| `Insufficient disk` | 输出空间小于估算 | 释放空间或换盘 |
| `Requires lerobot 0.5.0, got ...` | `LEROBOT_PY` 环境版本不对 | 安装/切换到 LeRobot 0.5.0 |
| `Output validation failed` | 输出 QC 未通过 | 查看 `qc_report.json` / 单元日志 |
| `plan.json` 无 jobs（退出码 2） | 输入不满足任何候选 | 看 `skipped` 原因 |

---

## 8. 验收口径与已知边界

- **全量验收**要求 809 个训练仓库全部覆盖、转换成功、verify/load 全部通过；
  用子集试跑时 `accept` 失败是设计使然。
- **`verification.passed=true` 不等于全量完成**：必须同时检查 `skipped`、
  `missing_expected_repos`、`corpus_complete`。
- 两个平台有 9 对仓库的集数/帧数不同（见 `ACCESS_AND_COVERAGE.md`）；
  每个仓库只能选一个平台的完整版本，不能混拼。
- 54 个仓库的 `state/action` names 不完整，包内用中性名称补全（见 `DIMENSION_REPAIRS.md`），
  未知通道不填入有物理含义的 55D 槽位，也不删除/截断。
- 62 个 HF `Cobot_Magic_*` 与 ModelScope `Agilex_Split_Aloha_*` 是同一仓库的别名，
  目录名两种都接受。
- 时间戳来自发布数据，不能当成实测采集时钟；`sample_dt_kind=1`、`mask_source_dt=0`。
- 夹爪保留原始尺度，不保证跨机器人可比；末端旋转只用明确声明的四元数，不猜 Euler。

---

## 9. `research/` 与 `reports/` 的定位

- **运行时只读** `research/` 中的 4 个文件：
  - `repos.json`：809 个训练仓库清单（覆盖检查）
  - `repository_aliases.json`：目录名 → canonical repo
  - `validated_schemas.json`：允许的 schema 白名单
  - `dimension_repairs.json`：54 个仓库的名称补全规则
- `research/metadata/`、`research/metadata_modelscope/`：逐仓库元数据快照（研究/审计用，运行时不读）。
- `reports/`：历史审计、性能、验收报告样本；正式验收以本次运行生成的
  `WORK/verification.json`、`WORK/official_load.json`、`WORK/acceptance_v4.json` 为准。

---

## 10. 最小操作清单

```bash
cd /mnt/sfs_turbo/liujin/code/blue_zone_code/data_preprocess/robocoin_v4_run

# 1) 编辑 config.sh：CONVERT_PY / LEROBOT_PY / INPUT / OUTPUT / WORK
# 2) 准备 INPUT（保持发布仓库名；每仓库一份完整 v2.1 版本）
# 3) 预检
bash run.sh plan

# 4) 查看 work/plan.json（skipped / missing / 磁盘）
python3 -m json.tool work/plan.json | less

# 5) 正式运行（建议后台）
nohup bash run.sh all > launcher.log 2>&1 < /dev/null &

# 6) 进度与验收
bash run.sh status
tail -f launcher.log
# 最终看 work/acceptance_v4.json 的 passed 是否为 true
```
