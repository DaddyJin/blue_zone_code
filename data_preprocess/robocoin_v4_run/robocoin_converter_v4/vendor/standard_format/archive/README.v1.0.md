# 统一数据格式规范(Unified 55D LeRobot v3.0)

> 版本:v1.0(2026-10-09)
> 本文档定义本团队所有机器人/ego 数据集的**最终统一格式**,并给出任一新数据集转换为该格式的标准流程、校验门槛与辅助脚本。
> 两条已验证的转换路径:
> - **EgoSuite(mcap)→ 本格式**:`hf_tools/egostandard_staging.py` + `unified_pipeline_v3`(生产线,4000+ 小时已按此交付)
> - **AgiBot(lerobot v2.1/v3 canonical_55d)→ 本格式**:`agibot_to_050/`(180 task 全量验证通过)
>
> 配套脚本统一归档在 `scripts/`,参考文档在 `docs/`。

---

## 0. TL;DR

- 格式 = **LeRobot v3.0 官方布局** + 55D 统一动作列 + 逐帧 `subtask_index` + `meta/subtasks.parquet` 词表 + `meta/episodes.jsonl` sidecar(子任务 source-of-truth,含 `skill`)
- 视频:960×728、h264/yuv420p、**30fps 容器**、~500MB 轮转、**单文件必须 ≤960s**(lerobot 0.5.0 float32 硬伤)
- 数据 parquet:~100MB 轮转,zstd
- 加载基线:**lerobot 0.5.0**,官方 `LeRobotDataset` 必须能加载并通过抽帧校验
- 交付判据:通过第 8 节校验链**全部门槛**;上传以 `_UPLOAD_COMPLETE.json` 为完成标记

---

## 1. 目录结构

最终交付单元 = **一个 task 一个目录,目录本身就是 LeRobot 数据集**(不存在 `canonical_55d` 之类的中间嵌套层):

```
<task_dir>/                                # 例: EgoStandard__Add_Sugar_and_Stir_Coffee
├── data/chunk-000/file-*.parquet          # ~100MB 轮转, zstd; 列规范见 §3
├── videos/observation.images.<cam>/chunk-000/file-*.mp4
│                                          # ~500MB 轮转; 单文件 ≤960s (durcap)
├── meta/
│   ├── info.json                          # codebase_version=v3.0, fps=30, features, custom
│   ├── tasks.parquet                      # 任务级语言指令词表 (task_index ↔ task)
│   ├── subtasks.parquet                   # 子任务词表 (subtask_index ↔ subtask)
│   ├── episodes.jsonl                     # sidecar: action_config(含 skill)/source_fps/...
│   └── episodes/chunk-000/file-*.parquet  # episode 索引: data/videos 的 chunk+file+时间戳定位
├── qc_report.json                         # L1 QC 报告
└── _UPLOAD_COMPLETE.json                  # 上传完成标记(本地+OBS 各一份), 见 §9
```

过程标记(不进交付、不上传):`.p2_done` `.remerged` `.durcap_done` `.retime_done` `.subtask050`。

命名约定:`<数据集名>__<任务名>`(如 `EgoStandard__...`、`EgoStandard-body__...`、`task_352`)。standard 与 body 子集靠前缀区分,可共存于同一 OBS 前缀下。

---

## 2. 坐标系与单位(跨数据集统一约定)

- 长度:**米**;角度:**弧度**;旋转:四元数
- **55D 内的四元数一律 xyzw**;透传列保留原始约定(egosuite 的 `observation.pose.*` 是 **wxyz** 世界系,`observation.extrinsics.*` 是 **xyzw**)——读透传列时务必看 `info.json.custom.coordinate_frames`
- **eef 位姿所在的坐标系是各数据集各自的身体/基座系,不是世界系**:
  - 摇操数据集(agibot/yam/droid):eef 相对**机器人基座系**(与底盘移动无关)
  - egosuite:eef(腕部)相对**逐帧躯干偏航系**(重力对齐 z;body 子集原点=pelvis、yaw=肩线减 90°;非 body 子集原点=head、yaw=头朝向代理,代理误差 ~5.8° 中位/~14° p95)
- 跨数据集一起训练时,统一靠**归一化**,不靠把各家坐标系换算成世界系——eef 都在"身体/基座"系下,语义天然对齐;世界系原始位姿在透传列里保留,需要时可自行重建

---

## 3. data parquet 列规范

### 3.1 通用索引列(所有数据集一致)

| 列 | dtype | 含义 |
|---|---|---|
| `frame_index` | int64 | episode 内帧号,0 起 |
| `episode_index` | int64 | episode 编号 |
| `timestamp` | float32 | **容器时间轴**秒数 = frame_index/30(见 §6 帧率语义) |
| `task_index` | int64 | 指向 `meta/tasks.parquet` 词表 |
| `subtask_index` | int64 | 指向 `meta/subtasks.parquet` 词表,**-1 = 未标注帧** |
| `index` | int64 | 全数据集全局行号(lerobot 官方要求) |

### 3.2 55D 统一动作列

五个列,均为 `fixed_size_list<float32>[55]`:

| 列 | 含义 |
|---|---|
| `state_unified` | 当前帧 55D 状态 |
| `action_unified` | **下一帧 state**(末帧复制自身) |
| `mask_state` / `mask_action` | 55 维 0/1:该维是否有真实映射(空槽=0) |
| `reliability` | 预留的置信度通道;**当前实现恒等于 mask**(以代码为准:空槽=0) |

**槽位布局**(`unified_pipeline_v3/unified_pipeline/schema/canonical.py`):

| 维 | 槽位 | 大小 | egosuite 映射 | agibot/yam/droid |
|---|---|---|---|---|
| 0:7 / 7:14 | left/right_arm_joint | 7+7 | 空(mask=0) | 关节角 |
| 14:21 / 21:28 | left/right_eef | 7+7 | **腕部**(手 21 关节的关节 0)xyz+quat xyzw,躯干偏航系 | 基座系 eef |
| 28:29 / 29:30 | left/right_gripper | 1+1 | 拇指尖–食指尖距离在 [0.04,0.10]m 归一化 clip 到 [0,1] | 夹爪开合 |
| 30:36 / 36:42 | left/right_hand | 6+6 | **预留**:当前 mask=0,等手部 retarget 后回填(见 §10.4) | 空 |
| 42:46 | waist | 4 | 空 | 腰部 |
| 46:48 | head | 2 | **头部**(yaw 相对躯干 + pitch) | 空 |
| 48:51 | mobile | 3 | 空 | 底盘(55D 不含底盘自由度时置空) |
| 51:55 | reserved | 4 | 空 | 空 |

### 3.3 透传列(passthrough)

**机制**:canonicalize 时,源 parquet 中原有、且不在 55D 表中的列,**原样追加**在 canonical 列之后。原则:**可以不处理,但不能丢**。

egosuite 实例(37 列,含 27 个透传列):

| 列 | 形状/dtype | 说明 |
|---|---|---|
| `source_timestamp` | float64 | **真实采集时钟**(墙钟),帧率还原唯一权威来源 |
| `observation.pose.head/headcam/right_eye_cam` | (7,) float32 | pos xyz + quat **wxyz**,**世界系** |
| `observation.pose.left_hand/right_hand` | (147,) = 21×7 | 手部 21 关节,关节顺序见 `info.json.custom.hand_joint_order` |
| `observation.pose.body`(body 子集) | (154,) = 22×7 | 身体 22 关节,顺序见 `custom.body_joint_order` |
| `observation.calibration.<cam>.{K,P,R}` | (9,)/(12,)/(9,) | 内参;**K/P 已 ×0.5 适配 960×728**,直接用即可 |
| `observation.extrinsics.<cam>` | (7,) | quat **xyzw** |
| `observation.intermediate.*` / `action.intermediate.*` | 各 (7,)/(2,) | 55D 映射的中间量(躯干系腕部位姿、pinch、头 yaw/pitch),便于审计 |
| `annotation.bad_frame.pose_hand` | bool | 坏帧标记 |
| `annotation.bad_frame.pose_hand.problem_types` | string(JSON) | 坏帧原因 |

1 维向量在 parquet 里存为标量 float32(lerobot 会把 shape-[1] feature 映射成标量)。

---

## 4. meta/ 规范

### 4.1 info.json

- `codebase_version: "v3.0"`,`fps: 30`
- `data_files_size_in_mb: 100`,`video_files_size_in_mb: 500`(必须与 §5 轮转一致)
- `features`:除官方列外必须注册 `"subtask_index": {"dtype": "int64", "shape": [1]}`;视频 feature 为 `{"dtype":"video","shape":[728,960,3],...}`
- `custom`(信息性,不参与加载):`converter`、`source`、`coordinate_frames`、`hand_joint_order`、`body_joint_order`、`mapping_55d`、`video_resize`、`time_axis`(帧率语义说明,见 §6)

### 4.2 词表文件(官方布局,两个同构)

- `meta/tasks.parquet`:列 `[task_index int64, task string]`,pandas index = `task`
- `meta/subtasks.parquet`:列 `[subtask_index int64, subtask string]`,pandas index = `subtask`;索引 **0..N-1 连续**,顺序 = 文本**首次出现顺序**(非字母序)
- pandas metadata 中 `"index_columns"` 必须正确,否则下游 `pq.read_table(...).to_pandas()` 行为不一致

### 4.3 episodes 索引(`meta/episodes/chunk-000/file-*.parquet`)

官方 v3 全量索引列:`data/chunk_index+file_index+file_from/to_offset`,每个视频相机的 `videos/<cam>/chunk_index+file_index+from/to_timestamp`。视频重合并/重切(§5)后**必须同步重写这些定位列**,tmp+rename 原子替换。

### 4.4 episodes.jsonl(sidecar,子任务的 source-of-truth)

每行一个 episode,egosuite 实例字段:

```json
{"episode_index": 0, "tasks": [...], "length": 13529,
 "action_config": [{"action_text": "...", "skill": null, "start_frame": 0, "end_frame": 541}, ...],
 "semantic_segments": ..., "session_metadata": ..., "source_mcap": "...",
 "source_fps": 25.0, "source_duration_s": 541.16, "torso_frame_src": "..."}
```

- `action_config`:`[{action_text, skill, start_frame, end_frame}]`,**episode 相对帧号,半开区间 [start,end)**
- **`skill` 只存在于这里**(agibot 有真实值如 `Wipe/Carry/TurnWheel`;egosuite 该键存在但目前为 null)。决策:**skill 不进 data parquet 逐帧列**;训练要用,按 `(episode_index, frame_index∈[start,end))` 关联 episodes.jsonl 取得
- `source_fps` / `source_duration_s`:原生帧率与真实时长,配合 `source_timestamp` 还原真实墙钟

---

## 5. 视频规范

| 项 | 值 |
|---|---|
| 分辨率 | **960×728**(源 1920×1456 精确 0.5×;`ffmpeg -vf scale=960:728`) |
| 编码 | libx264,crf=18,preset=fast,yuv420p,+faststart |
| 帧率 | 容器一律 **30fps**(原生 25fps 的处理见 §6) |
| 轮转 | ~500MB/文件,1000 文件/chunk |
| **单文件时长上限** | **≤960s 硬上限**,见下 |

**为什么是 960s**:lerobot 0.5.0 的 `decode_video_frames_torchcodec` 用 **float32** 比较查询时间戳与帧 pts,容差 1e-4。t>1024s 时相邻 float32 间隔 1.22e-4 > 容差 → `FrameTimestampError`,**必炸**;t<1024s 最坏 6.1e-5 必过。取 960s 留余量。切分**只在 episode 边界**(单集最大 ~711s,不切集内);≤960s 的文件硬链接进新树零成本,>960s 的用 PyAV 单输入多输出重打包(pts/dts 整数 tick 重基),末集尾部开放收容器尾包,帧数差 >10 才报错。脚本:`scripts/egostandard_durcap_rotate2.py`。

---

## 6. 时间轴与帧率语义(重要,踩过坑)

LeRobot v3 一个数据集只有一个 fps。我们的约定:

- **所有视频按 30fps 打包**,`timestamp` 列 = frame_index/30,**帧索引与 action_config 帧区间永远精确**
- 原生 **非 30fps**(egosuite 约 21% 的集是 25fps)的 episode:容器时间轴按比例压缩(25/30,等价于加速播放),**不重采样、不丢帧**
- 模型/下游要知道真实时间:用 `source_timestamp` 列(真实采集时钟)+ episodes.jsonl 的 `source_fps`/`source_duration_s`;`info.json.custom.time_axis` 里也有文字说明
- **后果自知**:非 30fps 的集在容器时间轴上被加速/减速。当前方案不显式给模型"变速标记",需要精确物理时间的下游(如速度估计)必须走 `source_timestamp`

---

## 7. 子任务标注规范(三种载体,互为冗余)

| 载体 | 位置 | 角色 |
|---|---|---|
| 逐帧 `subtask_index` | data parquet | 训练主通道;-1=未标注 |
| `meta/subtasks.parquet` | 词表 | index ↔ 文本 |
| `action_config` | meta/episodes.jsonl | **source-of-truth**;含 `skill`;帧边界半开区间 |

- 段边界一律 `[start_frame, end_frame)`;首段前/末段后的帧 = -1(egosuite 实测头/尾未标注普遍存在)
- 逐帧列必须由 action_config **构造性派生**(bisect_right 于真实时钟),并经过 `fix_subtask_bounds` 幂等校正(历史 bug:直接对 end_times bisect 会把边界帧错分给下一段)
- 词表只增不改;零宽段文本也入词表

---

## 8. 校验链(交付前每 task 必过,按序执行)

实现:`hf_tools/incr_validate_upload.py`(生产线),单项脚本在 `scripts/`。

| # | 检查 | 通过标准 |
|---|---|---|
| 1 | **drift 时间轴漂移** | 每相机 \|Σ视频容器时长 − Σ帧数/30\| ≤ **0.5s** |
| 2 | **durcap** | 所有 mp4 ≤960s;触发了重切则重切后**复跑 drift** |
| 3 | **subtask bounds** | action_config 与逐帧列逐帧一致(幂等修复,不判失败) |
| 4 | **lerobot 0.5.0 官方加载** | `LeRobotDataset` 加载成功;`meta.subtasks` 非 None;`__getitem__` 返回 `subtask` 字符串;随机 500 帧(seed=0)subtask 文本↔索引一致(**触发真实视频解码**);统计 -1 占比 |
| 5 | **stts 健康** | 每 mp4 异常 stts delta 样本占比 ≤**2%**(>2% 硬失败;≤2% 记 warning,权威判据是 #4) |
| 6 | (抽检)**帧内容配对** | 修复后视频帧 vs 源 mcap 包逐帧 MAE<2(`pairing_check2.py`) |

**drift 超标后的修复分级**(`incr_repair_loop.py` 自动化,MAX_ATTEMPTS=3):
- `uniform`(单一步长)→ `mp4_timescale_fix.py`(mdhd/elst 补丁,无损)
- `mixed`(多种合法网格步长)→ `mp4_retime_mixed.py --mode pts`(按容器 pts 重建呈现序)
- `garbage`(伪造 tick/ctts 离网格)→ `mp4_retime_mixed.py --mode poc`(H.264 POC 图像序重建)
- 修完复验 drift 归 0 才算完;3 次失败置 `repair_failed` 留人工

---

## 9. 上传与完成标记

- 上传工具:`obs_batch_upload.py`(`--skip-existing --workers 8`,凭证从 `data_pipeline/.env` 读)
- 目标前缀:`obs://egodata/egosuite_processed/v0/standard/`(egosuite 实例)
- **`_UPLOAD_COMPLETE.json`**:所有数据文件传完后**最后上传**,内容含 `file_count/total_bytes/files[{path,size}]/upload_finished` 与约定文字。**下游下载前必须看到此文件,下载后按 files 清单逐个核对 size**;无标记 = 未传完,不得下载
- 上传后本地副本会被清理脚本删除(源 mcap 与本地转换副本),OBS 为唯一留存——所以**未过校验链不得上传**

---

## 10. 新数据集接入指南(checklist)

1. **写 staging adapter**(参考 `scripts/egostandard_staging.py`):把源格式(mcap/rosbag/hdf5/…)转成**临时 lerobot v3.0 数据集**——视频转码 960×728@30fps、info.json/features/tasks/subtasks/episodes 全量生成、透传列全部保留、action_config 写 episodes.jsonl
   - 任务名含冒号等特殊字符时,用无冒号别名跑 pipeline 再改名(batch_robot 按 `name:path` 首个冒号切割)
2. **55D 映射**:确定源数据各通道落到哪个槽位;**没有对应槽的数据不要硬塞,放进透传列**;四元数统一到 xyzw;action = 下一帧 state
3. **跑 canonicalize + repack**(`unified_pipeline_v3/batch_robot.py`,`official_layout=True`):产 100MB data / 500MB video 轮转的官方布局
4. **回填类后处理**(如手部 retarget):读透传列 `observation.pose.*_hand` → retarget → 写入 hand 槽(30:36/36:42,或覆盖 gripper 槽 28:30)→ 同步把对应维 mask 置 1 → 更新 info.json custom
5. **视频重合并**(`egostandard_remerge_videos.py`)→ **durcap**(§5)→ **校验链**(§8)全过 → 上传 + `_UPLOAD_COMPLETE.json`
6. **并行度纪律**:磁盘 IO 是瓶颈(HDD ~96% util @ 6 并发),4–6 worker 封顶;单机 60GB 内存时大任务并行 ≤4(单 worker 峰值 >9GB RSS)

### 已验证范例
- **egosuite(mcap)**:`scripts/egostandard_staging.py`(adapter)+ 上述全流程
- **agibot(已是 lerobot)**:`scripts/convert_agibot_subtasks.py` 只补 subtask 三件套(逐帧列+词表+info.json 注册),episodes.jsonl 原样保留;~13.5 万帧/s/worker 级,全量 180 task ≈40 分钟;`scripts/validate_agibot_subtasks.py` 逐帧全量校验 + `--official` 官方加载

---

## 11. 已知坑(血泪清单)

1. **lerobot 0.5.0 float32 容差**:视频文件 >1024s 必炸(§5),durcap 是硬门槛不是可选项
2. **import 顺序**:lerobot 环境里**先 import lerobot 再 import pyarrow/pandas**,否则 torchcodec 动态库符号冲突;校验脚本全部 lazy import
3. **官方加载环境**:`lerobot050` env + `LD_LIBRARY_PATH=<envs/lerobot>/lib` + `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` + `HF_DATASETS_CACHE` 指大盘
4. **中断恢复**:data parquet 重写必须 tmp+rename 原子替换;半成品会留下**重复列名**,`pq.read_table` 的 dataset scanner 会 choke,要用 `pq.ParquetFile(f).read()` 按位置去重
5. **concat -c copy 拼接会污染容器时间轴**(400/1/0 tick 伪 stts),这就是 drift 的主要来源——所以校验链必须**在合并之后**跑,不能在 staging 阶段验完就完事
6. **`reliability` 当前 = mask**(不是恒 1),以代码为准
7. **batch_robot `--datasets name:path` 按冒号切割**,任务名含冒号必须用别名
8. **验证环境/生产环境分离**:forge_venv(av+pyarrow,无 pandas)、finevla(obs sdk+pandas)、lerobot050(官方加载)各司其职,别混用
9. subtasks.parquet 的 pandas index metadata 缺失或列序不对,下游 pyarrow/pandas 读取行为会不一致——校验脚本有严格比对
10. 非 30fps 源的"加速/减速"是**设计使然**不是 bug,下游有异议先指向 `source_timestamp`

---

## 12. scripts/ 清单

| 脚本 | 用途 |
|---|---|
| `convert_agibot_subtasks.py` | agibot 式 action_config → 050 三件套(逐帧列+词表+info.json) |
| `validate_agibot_subtasks.py` | 050 逐帧全量校验 + `--official` lerobot 0.5.0 加载 |
| `egostandard_050_official_load.py` | lerobot 0.5.0 官方加载校验(任何本格式数据集通用) |
| `egostandard_timeline_drift_scan.py` | drift 检测(0.5s 容差) |
| `egostandard_durcap_rotate2.py` | >960s 视频按 episode 边界重切 |
| `stts_census.py` | mp4 stts 健康普查(只读 moov,毫秒级/文件) |
| `egostandard_fix_subtask_bounds.py` | 以 action_config 为准幂等校正逐帧 subtask 边界 |
| `mp4_timescale_fix.py` / `mp4_retime_mixed.py` | drift 修复:timescale 补丁 / pts·poc 重排 |
| `egostandard_staging.py` | **参考 adapter**:EgoSuite mcap → 临时 v3.0(新数据集照此写 staging) |
| `egostandard_remerge_videos.py` | 逐集 mp4 → ~500MB 轮转会并 + episodes 索引重写 |
| `obs_marker.py` | `_UPLOAD_COMPLETE.json` 生成/校验 |

详细用法见 `scripts/README.md`;参考文档:`docs/egostandard_55d_README.md`(下游用户向)、`docs/EGOSUITE_PIPELINE_RUNBOOK.md`(生产 runbook)。
