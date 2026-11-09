# RoboCOIN → Unified55 / LeRobot v3

仅处理**已下载、已解压**的数据。正式输入输出由调用者指定；不会下载、解压或修改原数据。29 仅用于样例调试。

## 范围与结果（v4）

- 目录快照共 **809 个训练仓库**。ModelScope 的 809 份元数据均可公开读取；合并两轮探测，809 个首个 Parquet 文件头均可读取。此处不等于全文件完整性验收。
- 62 个 HF `Cobot_Magic_*` 仓库在 ModelScope 上叫 `Agilex_Split_Aloha_*`。转换器同时接受两种目录名，并识别为同一仓库；完整映射在 `research/repository_aliases.json`。
- **809/809 元数据匹配到已测试的 26 种数值结构**（24 种常规结构、2 种明确名称修复规则）。不再因此前缺少结构样例而排除仓库。
- 实际转换与保真验收：**29 个代表仓库、58 集、29,064 行**，包含真实 ModelScope 别名目录；另有 LeRobot 0.5.0 加载报告。没有在 29 全量处理 809 个仓库。
- 54 个仓库的 names 缺漏修复保留全部原值，见 `DIMENSION_REPAIRS.md`。9 对 HF / ModelScope 仓库的集数或帧数不同，见 `ACCESS_AND_COVERAGE.md`；必须选择一份完整版本，不能拼混两站文件。

## 转换规则

| 内容 | 处理方式 |
|---|---|
| 输出单位 | 每个来源仓库、每个原始 split 单独一个 LeRobot v3 数据集；不跨机器人混拼 |
| 视频 | 保留原分辨率、fps、相机名；v2.1 episode 视频按编码参数分组，无损拼接；默认每文件约 128 MB，单个超大 episode 不强切 |
| 不一致视频 | 同相机分辨率/fps变化、帧数与表不符、坏帧或时间轴异常会令该任务失败，其余任务继续；不偷偷丢帧或重编码 |
| state / action | state 映射当前观测；action 默认保留发布值。只有整集动态且所有已映射字段、共享具名原始分量逐值完全相同，才改为下一帧 state 目标；末帧 action 全 0、mask 全 0 |
| 55D | `state_unified`、`action_unified`；另有 `mask_state`、`mask_action`，缺失项置 0、mask=0 |
| 关节 | 按名称与顺序填入左右臂、可容纳的手指、腰部、头颈。手指超过 6 维等不适配情况保留完整原列，不任意截断 |
| 夹爪 | 保留真实数值尺度；按作者 PR #7 的原字段定义和发布标注交叉核验取“越大越开”。实际有约 0～100、0～1000、角度量等，**不裁剪、不强制归一化**；单位记录为 source_native |
| 末端 | 30 个 AgiBot 来源使用原生 SDK 的 xyz+xyzw，位置和旋转来自同一列、同一原生坐标系；有限且范数 >1e-8 的四元数归一化，不设 1% 范数门限；归一化后孤立大角度跳变仍屏蔽，保持集内符号连续。其余有仿真列的来源只映射仿真 xyz；未证明的 Euler 约定不猜测 |
| 坐标系 | 按来源记录仿真坐标系、机器人 owner；作者只声明基座/两脚中心及 x前/y左/z上，不能据此当作跨机器人共享世界原点 |
| 时间 | 保留原 timestamp，生成 `sample_dt` 与有效 mask。现有证据仅支持发布数据的相邻时间差，`sample_dt_kind=1`；不伪称实测采集时间，`mask_source_dt=0`、`mask_action_dt=0` |
| 语言 | task 字典保留；有 subtask 字典时，按作者加载器规则依序拼接非 null 标签，生成 `[start_frame,end_frame)` 的 action_config；原 ID 与字典仍保留 |
| 原数据 | 全部原 parquet 列进入 `source.*`，原 meta、annotations、README/info.yaml 另存于输出 meta 下 |
| split | 严格按半开区间解释；没被原 split 覆盖的 episode 放入 `unassigned_*` 单元，不能默认为训练集 |

55D 槽位（从 0 开始）：左右臂 0:7 / 7:14；左右 EEF 14:21 / 21:28（xyz+xyzw）；夹爪 28/29；左右手 30:36 / 36:42；腰 42:46；头 46:48；底盘 48:51；51:55 保留。具体每列来源、单位、frame、未映射项写在 `meta/input_manifest.json`。

注意：作者说明 action 可能来自主臂命令，也可能由 state 派生，未逐仓库明确细分。本工具不把保留的发布 action 统称“下一帧实测状态”；逐集转换决策见下节。夹爪方向依据作者字段定义及首尾样例核验，未做实体机器人标定。8 个 Galaxea 库的 action 夹爪标注与原向量不一致，不能用标注覆盖原 action。

## v4 动作来源与旋转有效性

- `action_source_kind=0`：保留作者发布 action。此标签不保证它一定是原始控制命令，作者也可能派生过它。
- `action_source_kind=1`：经整集检查满足同帧复制规则，由下一帧 state 派生。`action_target_step=1` 表示目标位于下一行；末帧置 0 且全部动作 mask=0。
- `meta/episodes.jsonl` 的 `action_semantics` 记录逐集原因。同一仓库不同集可采用不同策略；静止集、少于 3 帧、仅部分字段相同、任一被比较值非有限都保留发布 action。逐值相同是转换判据，并非作者处理历史的证明。
- 原始列全部保留在 `source.*`。不要再给 `action_unified` 统一移帧，也不要将“原生控制目标”和“下一帧实测位置”当作完全等价；训练可按来源选择。
- 派生目标的时间跨度使用 `sample_dt`，但目前发布 timestamp 未证实为实测时钟，不能把 `mask_source_dt` 或 `mask_action_dt` 强行改成有效。
- 已知 xyzw/wxyz 可以重排、归一化。当前旋转恢复只用于元数据明确的 SDK 四元数；不猜测仿真 Euler 顺序/角度单位。xyz 已确认、旋转未确认时仅屏蔽旋转。
- 旋转质检：无穷/NaN、范数 ≤1e-8 无效；归一化后若中间一帧与前后均相差 >120°、而前后相差 <30°，屏蔽该孤立旋转。该门限是显式的保守启发式，不是作者提供的机器人速度限制；不插值伪造角度，xyz 不受影响。
- 已发现 AgiBot 两个仓库的夹爪命令与观测有较大、非统一的时间偏移，详见 `SEMANTICS_VERIFICATION.md`；不自动反转或平移。
- v4 必须使用新的 OUTPUT 和 WORK；v3 结果不可混入或当作 v4 断点继续。v4 的相同输入、相同代码可正常断点校验。

## 正式机安装与执行

转换环境在 29 的 Linux ARM64 / Python 3.10 上实测。转换本身无需 PyTorch、LeRobot、GPU或网络；环境安装阶段需预先准备依赖。

```bash
cd /path/to/robocoin_converter
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# INPUT 下允许多层目录；每个仓库目录保留 HF 或 ModelScope 的仓库名。
# 每个仓库应有 meta/、data/、videos/；不要把同仓库的两站副本都放入 INPUT。
# 三个目录必须互不包含；INPUT 必须在整个处理期间保持不变。
INPUT=/data/robocoin_extracted
OUTPUT=/data/robocoin_unified55_v4
WORK=/data/robocoin_conversion_work_v4

.venv/bin/python scripts/convert_repos.py plan \
  --input "$INPUT" --output "$OUTPUT" --work "$WORK"

# 先查看 WORK/plan.json 的跳过项、计划任务和磁盘预算。
.venv/bin/python scripts/convert_repos.py convert \
  --input "$INPUT" --output "$OUTPUT" --work "$WORK" \
  --workers 4 --video-workers 4 --decode-threads 8 --cpu-budget 128

.venv/bin/python scripts/verify.py \
  --plan "$WORK/plan.json" --output "$OUTPUT" --report "$WORK/verification.json"
```

`workers` 是仓库/split 进程数，`video-workers` 是每个进程并行的视频数，`decode-threads` 是每个视频的解码线程数；三者乘积不能超过 `cpu-budget`。程序限定 CPU affinity，原有 affinity 少于 128 时必须相应调小参数。

建议保留 256 GB 内存预算；这不是强制内存上限，实际峰值与 episode 长度和原始列宽有关。磁盘预检保守估算**额外可用空间**为 `2 × 有效输入字节 + 4096 × 行数 + 2 GiB`；输入本身需另占空间。不会删除输入；该估算也不包含额外性能测试副本。

## 断点恢复与验收

- 相同代码、输入与参数下重新执行同一命令：对已完成结果核对完整 SHA256 后跳过。
- 未完成单元在隐藏 pending 目录重做；最终目录在数据、元信息和哈希清单齐备后才发布。恢复粒度是仓库/split，不是视频内部。
- 代码/输入/解码参数变化时拒绝沿用旧结果，应换输出版本。不要在任务运行中改代码。
- 缺失必要文件、未验证结构、未获明确修复规则支持的 shape/names 不符等记入 `plan.json.skipped`；运行异常记入 `progress.json`，其他任务继续。
- `verify.py` 核对所有输出 SHA256、每一行原列、全部 55D 已映射槽位和 mask、时间差、全部压缩视频 packet 顺序和内容，并实际读取 RGB。
- `verification.json.passed=true` 仅代表计划内输出通过；还必须审阅 `skipped` 和 `missing_expected_repos`。样例或部分语料的 `corpus_complete=false` 是正常结果，不能忽略它而宣称全量完整。
- 质检覆盖结构、数值保真、时间对齐和可解码性，不等于证明所有动作标注和物理单位均正确。

**LeRobot 0.5.0 实际加载测试单独交付**于 `robocoin_loader_tests` 包。它使用官方 LeRobot 类的薄兼容子类，处理缺失 subtask=-1 和原始 singleton-list 列；不是未经适配的原生 LeRobotDataset。测试含每集拼接边界与 RGB 逐像素比较，详见该包 README。

## 性能调优

原 v0 的 21 仓库 / 42 集基准样例约 756 MiB（未计入本次新增两个样例和名称修复扫描的开销）。完整流程包含拼接、全帧解码检查、数值转换及哈希；测得 16×4×1 约 47.5 秒，4×4×8 约 19.6 秒。最终复测与进程树 RSS 记录见 `reports/performance_final.json`。这不是全量吞吐承诺，也不包含另行执行的 verify/LeRobot 加载验收。

在正式机有不同磁盘、CPU或长 episode 时，用**代表性小子集**重测，勿直接拿全部数据跑多个副本：

```bash
.venv/bin/python scripts/benchmark.py \
  --input /data/representative_subset \
  --output /data/benchmark_outputs --work /data/benchmark_work \
  --report /data/benchmark_report.json \
  --configs 16x4x2,8x4x4,4x4x8
```

基准目录必须是新目录。全量时间应根据正式机同分布样例的耗时和行数估算，并给最终验收另留时间；当前小样例不能可靠推出全部 809 仓库的完成时间。

## 文件与证据

- `scripts/convert_repos.py`：预检、并发转换、恢复、输出清单。
- `scripts/profile.py`：按真实字段名映射 55D；`repair_schema.py`：按明确规则校验并修复缺名；`enrich_annotations.py`：语言；`fast_media.py`：无损视频路径。
- `scripts/verify.py` / `benchmark.py`：保真验收 / 正式机实测调参。
- `vendor/standard_format/`：随包冻结的标准工具，基础版本 `2.1-robocoin`，另含本次 v4 语义补丁；不要和其他任务的旧工具混用。
- `research/`：仓库清单、元数据快照、已验证结构清单。
- `reports/`：样例、性能、原始数值/视频校验及官方加载结果；本次以 `verification_semantics_v4.json`、`official_load_semantics_v4.json`、`catalog_coverage_semantics_v4.json`、`dimension_repair_validation.json` 为准；其中 29 的绝对路径只用于审计，不作为正式机参数。
- [作者字段说明](https://github.com/FlagOpen/RoboCOIN#lerobot-features-explanation)；语言拼接依据其 [加载器实现](https://github.com/FlagOpen/RoboCOIN/blob/b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd/src/lerobot/datasets/lerobot_dataset.py#L751)。

已有旧输出不要混入 v4 输出目录。只恢复 54 个缺名仓库仍可用 `--only-repairable` 写新目录；补做其他仓库时，将待补仓库作为单独 INPUT，原目录名保持不变。详见 `DIMENSION_REPAIRS.md`。

## 核心监督语义（v4）

臂关节为必备表示，左右臂各 6 或 7 个关节，单位 rad；灵巧手不伪造夹爪值，完整手指关节透传。末端监督需相应 7 维 mask 全部有效，不能将只有 xyz 的行当成完整位姿。

本版在 809 库首尾 1,618 集、1,153,566 行数值样例上通过核验：779 集满足整集精确复制规则并派生下一状态目标，839 集保留发布 action。这个数量是样例结果；正式转换会逐集重新判定，不使用仓库白名单。

正式机可在已解压输入上执行数值审计（只读取 Parquet，不改数据）：

```bash
.venv/bin/python scripts/audit_numeric_semantics.py \
  --input "$INPUT" --report "$WORK/semantic_audit.json" --workers 8
```

这会扫描实际输入中的全部本地 Parquet，逐集记录臂关节及夹爪的时间对应、夹爪方向交叉检查、原生 SDK 四元数范数。样例统计不能替代正式机全量检查。详见 `SEMANTICS_VERIFICATION.md`。
