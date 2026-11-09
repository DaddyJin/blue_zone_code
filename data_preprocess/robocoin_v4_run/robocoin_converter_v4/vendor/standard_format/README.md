> 本交付包是 GM100 专用 `2.1-robocoin` 扩展，公共标准原文件未改动。下文旧版“夹爪必须0–1”在本包中由显式 `native_open_direction` 映射覆盖：原生尺度保留、只统一为越大越开，具体尺度/变换见每个数据集info.custom字段。新增sample_dt/kind区分发布时间网格与真实时钟；source_dt只接受实测时钟。本包的通用converter已针对GM100做扩展，非GM100请使用原公共标准工具。项目根README.md和CLOCKS.md优先说明本包行为。

# Unified55 数据格式与接入规范

版本 **2.0 · 2026-10-20**。交付格式为 **LeRobot v3 目录布局 + 固定 55D 列 + 明确的数据语义和时间来源**。适用于机器人、ego、仅观测和纯视频数据。只统一结构，不假定不同来源的控制量、坐标系和夹爪已经等价。

## 1 接入流程与工具

1. **检查源数据**：确认机器人、状态／动作含义、坐标系、夹爪、采样时钟及单个数据集内的视频规格。
2. **选配置**：从 [`examples/`](examples/) 复制机器人、源动作、ego 或纯视频示例；按真实字段修改映射。
3. **准备 manifest**：每个 episode 指向源 Parquet／视频，可引用共享文件中的区间；具体原始格式用 [`adapter_template.py`](tools/adapter_template.py) 接入。
4. **转换、校验、实际读取**：通过后生成 `SUCCESS.json`；半成品不交付。

```bash
python -m pip install -r tools/requirements.txt
python tools/probe_videos.py /path/to/source/videos --output inventory.json
python tools/convert.py manifest.json /path/to/new_output
python tools/validate.py /path/to/new_output --decode --samples 8
python tools/smoke_load.py /path/to/new_output
```

已有 LeRobot：`python tools/from_lerobot.py SOURCE profile.json manifest.json`，仍需自行填写状态／动作、时钟和坐标语义。纯视频可直接用 `make_video_manifest.py`。完整参数见 [`tools/README.md`](tools/README.md)。

## 2 交付目录

一个交付单元 = **同一机器人／主体、同一语义 profile、同一原始数据划分**。可以按任务或来源分片，不能混入含义不同却未区分的控制量。同相机在该单元内使用一致的分辨率和标称 FPS；不同单元可以不同。

```text
<dataset>/
├── data/chunk-000/file-000.parquet
├── videos/observation.images.<cam>/chunk-000/file-000.mp4  # 有视频时
├── meta/
│   ├── info.json                         # v3 features 与语义登记
│   ├── stats.json                        # 有效分量统计
│   ├── tasks.parquet                     # task_index ↔ task
│   ├── subtasks.parquet                  # subtask_index ↔ subtask
│   ├── episodes/chunk-000/file-000.parquet # 官方行区间、视频区间索引
│   ├── episodes.jsonl                    # 每集来源、时间与标注
│   └── video_timing/                     # 有视频时：实际 PTS 和帧定位
├── qc_report.json
└── SUCCESS.json                          # 完成本地转换与验收
```

上传／云端完成标记是分发层约定，不代替本地验收；本规范不要求使用特定云服务。

## 3 每行必须具有的列

所有数据集均保留下表列，包括无状态／动作的数据。1 行对应一个观测样本；有多相机时先在 adapter 中对齐，不能靠改 FPS 字段解决异步采样。

| 列 | dtype／形状 | 填写规则 |
|---|---|---|
| `index`、`episode_index`、`frame_index` | int64 标量 | 全局行号连续；集编号连续；集内帧号从 0 开始 |
| `timestamp` | float64 标量 | 集内视频查询时间；无视频时为集内样本时间。不是“真实时间”的默认依据 |
| `state_unified`、`action_unified` | fixed_size_list<float32>[55] | 按 §4 映射；缺失槽填 0 |
| `mask_state`、`mask_action` | 同上，元素 0／1 | 该分量可用于监督；缺失／未知／质量不合格置 0 |
| `reliability` | 同上 | 本版等于 `mask_state`，不是校准后的置信度 |
| `source_timestamp` | float64 标量 | 原始采样时钟，工具转为相对侧车所记原点的秒数；未知填 0 |
| `mask_source_timestamp` | float32 标量，0／1 | 是否有可用源时钟数值；其可信等级见每集 `source_time.kind` |
| `source_dt`、`mask_source_dt` | float64／float32 标量 | 当前到下一观测的**真实采样间隔**及有效性；仅 `measured` 可启用；未知、异常、末帧均为 0／0 |
| `action_dt`、`mask_action_dt` | float64／float32 标量 | 动作目标的真实时间跨度；未知为 0／0；不默认等于 `1/fps` |
| `task_index`、`subtask_index` | int64 标量 | task 必须命中词表；无任务用空字符串词条；无子任务用 -1 |
| `mask_language` | bool 标量 | 当前任务文本非空才为 true |
| `qc_sample_valid` | bool 标量 | 整个样本是否可用；false 时两个 55D mask 均为 0 |

原始信息**必须可回溯且不丢弃**：优先透传为 `source.<原列名>`；大型／嵌套内容可保留在可访问的原始文件中，通过每集 `source_data`、`source_columns` 和来源版本定位。不能映射到 55D 的量单独保留，不能硬塞进空槽。

## 4 55D 如何填写

索引使用 Python 半开区间。**没有要求每个槽都填，也没有要求关节和末端二选一。**

| 槽位 | 大小 | 可填内容 | 无对应数据时 |
|---|---:|---|---|
| `0:7`／`7:14` | 各 7 | 左／右臂关节通道；按真实关节顺序登记，6 轴只填 6 维 | 0，mask=0 |
| `14:21`／`21:28` | 各 7 | 左／右末端或腕部 `[x,y,z,qx,qy,qz,qw]` | 只知道 xyz 可只启用位置；无旋转则四元数 mask 全 0 |
| `28:29`／`29:30` | 各 1 | 左／右开合量或明确登记的 pinch 代理；有效值 0=闭、1=开 | 0，mask=0 |
| `30:36`／`36:42` | 各 6 | 已有明确 6 维定义的手部关节／retarget 结果 | 不把 21 个手关键点强压入此槽 |
| `42:46` | 4 | 腰部通道 | 逐维登记物理量和单位 |
| `46:48` | 2 | 头部 yaw／pitch | 登记所相对的参考系 |
| `48:51` | 3 | 底盘通道 | 明确区分 `[x,y,yaw]` 与 `[vx,vy,wz]` |
| `51:55` | 4 | 保留 | 必须为 0，mask=0 |

每个映射必须登记 **源列、槽位、所属主体 owner、物理量 quantity、单位 unit、来源 provenance**；关节再登记有序 `component_names`，末端再登记 `frame`、`reference_point`、四元数顺序。用米、弧度、xyzw；速度／增量必须写明，不能伪装成绝对位置。未知值或人为常量不因为“有数值”就启用 mask。

### 动作必须选择一种策略

| `action.mode` | 适用条件 | 规则 |
|---|---|---|
| `next_state` | 用未来状态作目标 | `action[t]=state[t+1]`，目标分量的有效性跟随下一状态；不跨集；末帧复制自身但动作 mask 全 0 |
| `source` | 保留源数据真实动作命令 | 独立映射，可与 state 不同；声明绝对／增量／速度、参考系和末帧是否仍有效 |
| `none` | 无动作数据 | action 和 mask_action 全 0；不能拿未知量伪造动作 |

纯视频：`data_kind=video_only`，五个 55D 列**全部为零**。有状态但无动作：保留 state，动作采用 `none`。原始 `observation.state` 若属于别的机器人，不得当作本单元的关节状态；OXE 的增强机器人应使用该增强机器人的字段。

夹爪的统一仅指开合方向。二值、连续宽度归一化、命令和人体代理需要在 profile 中分别说明；同为 0.5 不保证同一物理宽度。坐标系、动作和夹爪选择见 [`docs/DECISIONS.md`](docs/DECISIONS.md)。

## 5 视频与时间

| 源数据情况 | 处理 |
|---|---|
| 单个数据集内同相机规格一致，类似 OXE 各小仓库 | `native`：保留原分辨率、标称 FPS、编码；已有 v3 合并视频可直接复用；逐集视频按码流兼容性无损合并 |
| 同一单元不同 episode 的分辨率／FPS 不同，类似 EgoSuite | `normalize`：显式选择该单元的尺寸和 FPS，例如 960×728、30fps；本工具逐帧保留数量并重定时，不自动补／丢帧 |
| 有状态／动作频率高于视频或多相机异步 | adapter 先按已知时钟关联或重采样；保留原始时间。状态位置、旋转、开关命令不能一概用同种插值 |

统一规格只针对一个交付单元。不同相机可以有不同尺寸；本工具要求每行都有已对齐的各相机帧，不能直接导入帧数不同的异步流。

**EgoSuite 25fps → 30fps 的保帧方案**：同一组 N 帧仍是 N 行，视频时间步长为 `1/30`；源真实间隔仍由采集时钟得到，例如 `0.04s`。播放速度会改变，物理时间不会被改写。若要保持播放时长并补帧／丢帧，应另做显式重采样 adapter，同步重建动作和标注，不能只对视频使用 fps 滤镜。

`source_time.kind` 取 `measured`／`published`／`nominal`／`video_pts`／`unknown`。仅有发布者的规则时间戳或 FPS，不能声称已知真实采集时钟：保留这些源时间，但 `source_dt`、`mask_source_dt` 为 0。OXE 此类发布时戳的差值可另列估计量，不能冒充实测间隔。

视频索引与实际 float64 PTS 必须匹配。**没有通用 960 秒硬限制**；长片应使用随附精确 PTS 读取器或验证过的后端。若必须兼容存在 float32 时间精度问题的旧后端，可按集边界限长。尺寸变换后，若有 K/P 内参／投影矩阵，必须同步变换；工具通过 `calibration_columns` 回填新列，原列可透传。

## 6 元数据必填与可选项

| 部分 | 要求 |
|---|---|
| `info.json` | **必须**：官方 v3 路径、features、FPS、帧／集计数；`custom.standard_format_version=2.0`、data_kind、profile_id、来源、55D 映射、action 策略、参考系及时间规则 |
| `stats.json` | **必须**：55D 有效分量统计；不能把缺失占位零算进均值。图像统计未计算时明确登记，不填伪造常量 |
| tasks／subtasks 词表 | **必须**：ID 连续，文本为 str，pandas index 为 task／subtask；空子任务词表允许，逐帧用 -1 |
| 官方 episode 索引 | **必须**：数据行区间，以及每相机文件／时间区间；拼接或轮转后同步更新 |
| `episodes.jsonl` | **必须**：集 ID、长度、来源集 ID、profile_id、原始 split、split_group_uid、source_time；有标注时填 action_config |
| `action_config` | 无标注用 `[]`；有则填 action_text、skill（未知 null）、集内 `[start_frame,end_frame)`；禁止把任务级描述伪称精细子任务 |
| `video_timing/` | 有视频时**必须**：每文件真实 PTS `.npy`；episodes.parquet 记录每集每相机的帧起点、长度和 PTS 路径 |
| 校准、原始手／身体关键点、原控制命令等 | **按源数据保留**，不强制源数据本来没有的内容 |
| 上传完成标记 | **可选分发层要求**，与本地 SUCCESS 分开 |

## 7 交付检查

必须检查：55D 类型和 mask、单位／四元数、集边界、真实时间来源、行与视频配对、索引与词表、子任务区间、视频完整解码，以及最终读取器真实抽帧。工具会检查结构和已声明规则；开发者仍需核实映射字段的物理含义。

同一原演示的分片、不同相机、原版与增强版应使用同一 `split_group_uid`。先按组划分再混训，不能随机逐帧拆训练／测试。

[`tools/selftest.py`](tools/selftest.py) 只生成小型合成数据，可在安装依赖后运行。旧 EgoSuite 固定 30fps 的脚本在 `scripts/`，旧规范在 `archive/README.v1.0.md`；它们是既有流程参考，不是本版通用约束。已交付 OXE 未被此次更新重写，其既有 `action_dt` 来自发布时戳差值，不能按本版实测时间解释；升级 v2.0 须重建相应时间列／mask，并补齐元数据。
