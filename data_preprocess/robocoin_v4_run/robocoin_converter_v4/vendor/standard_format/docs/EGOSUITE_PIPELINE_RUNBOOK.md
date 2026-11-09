> **历史流程（v1）**：本文保留已有 EgoSuite 的处理约定；固定尺寸、30fps、时长及特定目录要求不再是通用约束。新数据集请从 [Unified55 v2 主规范](../README.md) 和 [通用工具](../tools/README.md) 开始。

# EgoSuite (EgoStandard) 转换流水线 Runbook

最后更新: 2026-09-14。本文档固化从 mcap 源数据到可交付 LeRobot v3 的**全部已确定细节**,
任何一步变更都必须同步更新本文档。新批次数据的转换先读 skill `egosuite-convert`
(`/root/.kimi-code/skills/egosuite-convert/SKILL.md`), 本文档是其现场补充。

## 0. 数据总盘(实测, 非估算)

- 源: `/mnt/disk1/EgoStandard/{EgoStandard,EgoStandard-body}/mcap/<task>/*.mcap`
- **3621 个任务 / 134,323 个 mcap / 4053.0 小时**(mcap summary 时间戳实测,
  `hf_tools/egosuite_duration_report.json`, per-task 明细 `egosuite_duration_by_task.json`)
- standard 1356 任务 3056.5h;body 2265 任务 996.5h
- standard 只是全集的字母序前缀(到 "Organize Clothes Outside Wardrobe"), 官方库还有未下载部分

## 1. 转换(p1 + p2)

- p1 首批: `egostandard_formal_driver.py`, **每任务只转前 `k=max(1, n//5)` 个 episode**
  (SAMPLE_EVERY=5, 连续前缀, 不许随机采样——p2 靠 `--skip-first k` 接着转)
- p2 续转: `egostandard_formal_driver_p2.py`, 转剩余 80% 并 merge:
  - 断点安全: `.p2_done` 标记 + `.p2_journal` 崩溃回滚
  - `merge_one` (egostandard_merge_p2.py): **旧视频文件不动**, 新 episode 视频 concat -c copy
    成 ~500MB 轮转文件续编号; task_index/subtask_index 词表只增不改(旧索引稳定);
    episodes parquet 重建(旧行原样), info.json 更新
- 单集任务(n=1)在 p1 即转完, 无 `.p2_done`——**完整性判定的唯一标准:
  info.json total_episodes >= 源 mcap 文件数**, 不要只看标记
- 转换参数: 30fps 时间网格, 视频 960×728(等比), preset=fast(用户选定, CPU 编码是瓶颈)
- **帧率归一**: 原生 25fps/36fps 的 episode 被归一到 30fps 网格(会加速/减速);
  `source_timestamp` 列保留原生时间戳, 真实时长用它算

## 2. 已确定的质量问题与修复(顺序不可乱)

### 2.1 视频时间轴漂移(三类, 现场重分类不盲信普查)
- 普查: `egostandard_ctts_census.py` → `ctts_census_report.json`
  (uniform 395 / mixed 83 / garbage 710, 共 1188)
- 修复(`egostandard_timeline_repair_driver*.py`, 幂等 `.retime_done` 标记):
  - uniform: `mp4_timescale_fix.py --write`(mdhd timescale 补丁, 含 elst 补丁, 幂等)
  - mixed: `mp4_retime_mixed.py --mode pts`(PyAV 容器重写, 呈现序=pts 排序)
  - garbage: `mp4_retime_mixed.py --mode poc`(H.264 POC 解析重建)
- 修完逐任务复跑 `check_task`(egostandard_timeline_drift_scan.py), worst_dev=0 才落标记
- **普查是时间快照**: p2 之后 merge 的任务会产生新漂移(concat -c copy 保留各
  episode 原始时间戳, 拼接后容器时长膨胀), 不会出现在旧普查里。**每批 p2 完成后
  必须对该批重扫漂移**; 上传前的全量复扫是最后一道闸(2026-09-11 曾拦下 55 个
  p2 晚合并的漂移任务, 全部补修通过)
- **坑**: PyAV remux 必须 `add_stream_from_template`(`add_stream("h264")` 写 1 字节包);
  pkill -f 会匹配到自己 shell(先单独跑 pkill)

### 2.1b concat 内部 pts 损伤(stts 快扫, 漂移扫描的盲区)
- p2 concat 合并可能产生**文件内部 pts 垃圾化但总时长与帧数恰好抵消**的文件,
  漂移复扫查不出来。2026-09-14 durcap 帧数校验抓出 22 文件/11 任务。
- 快扫: `stts_census.py`(只读 moov 的 stts 盒, 毫秒级/文件, 可全量) → 清单 `stts_bad22_files.json`
- 修复: `mp4_retime_mixed.py --mode poc` 重建; 修后 stts 复扫清零 + 集边界对齐 + 配对验证
- **p2 之后每批任务处理流程固定为: stts 快扫 → 漂移扫描 → 修复 → fix_subtask_bounds
  → durcap → 0.5.0 抽检 → 追加上传**

### 2.2 子任务边界修复
- `egostandard_fix_subtask_bounds.py`, 幂等, 对全部 3621 任务跑过一轮;
  p2 每次 merge 后要重跑(merge 保留旧 subtask_index, 理论安全, 跑一次保险)

### 2.3 视频文件时长上限(lerobot 0.5.0 torchcodec 容差硬伤)
- lerobot 0.5.0 `decode_video_frames_torchcodec` 用 **float32** 比较查询时间戳与帧 pts,
  容差 1e-4。t>1024s 时相邻 float32 间隔 1.22e-4 > 容差 → `FrameTimestampError`;
  t<1024s 时最坏 6.1e-5 必过。**视频文件必须 <1024s**(默认容差下官方加载才稳)。
- 修复: `egostandard_durcap_rotate2.py`(v2, 只 PyAV 重打包 >960s 文件, ≤960s 硬链接不动,
  IO 5.4TB→2.4TB; v1 全量重写已弃用)。批量 `egostandard_durcap_driver.py`。
  按 episode 边界重打包为 <=960s(episode 最大 711s, 不切集内),
  PyAV 整数 tick 重基 pts/dts, 每新文件 ffprobe 帧数校验(兜底, 曾抓出 stts 受损文件),
  重写 episodes parquet 视频定位列, 幂等 `.durcap_done`
- 验证链: 重轮转 → check_task 漂移 → 官方 0.5.0 加载(必须含视频解码抽帧) → 配对
- **p2 新产出的任务也要过这一关**; standard 集 p2 补转完成后同样处理

### 2.4 验证(上传前必须全绿)
- 漂移复扫全零
- 帧内容配对: `pairing_check2.py`(旧 pairing_check.py 有管道死锁, 禁用)——
  修复后视频帧 vs mcap 原始包逐帧 MAE<2(960×728 重编码正常损耗)
- 官方 0.5.0 加载: `egostandard_050_official_load.py`, 环境:
  `LD_LIBRARY_PATH=/home/z00827204/miniconda3/envs/lerobot/lib HF_HUB_OFFLINE=1
  HF_DATASETS_OFFLINE=1 /home/z00827204/miniconda3/envs/lerobot050/bin/python`

## 3. 透传列(保留未处理, 下游自行取用)

parquet 中除 55D 动作槽外的 mcap 信息全部原样保留:
`observation.pose.*`(head/hands/body…)、`observation.calibration.*`、`observation.extrinsics.*`、
`observation.intermediate.*`、`action.intermediate.*`、`annotation.bad_frame.*`、
`source_timestamp`、`subtask_index`(逐帧子任务标注, 词表在 meta/subtasks.parquet)。
body 任务比 standard 多 `observation.pose.body/lower_body/upper_body` 及对应 bad_frame 列——
**这是程序内区分 body/standard 的依据**(目录名前缀亦可)。
手部数据未做 retarget, 同事取 `observation.pose.left_hand/right_hand` 自行处理后回填 55D 槽。

## 4. 上传

- 工具: `data_pipeline/scripts/obs_batch_upload.py`(finevla python, 凭证自动读 data_pipeline/.env)
- 目标: `obs://egodata/egosuite_processed/v0/standard/`(standard+body 都放这, 已确认不冲突)
- 参数: `--keep-dirname --skip-existing --workers 8`, setsid nohup 后台 + task_monitor 条目
- demo(10 任务)在 `obs://egodata/egosuite_demo/v1/`, 已交付验证

## 5. 监控

- 页面: task_monitor, 端口 8080, `config.json` 加条目即生效
- 总盘进度: `egosuite_progress_daemon.py` 每 300s 写 `task_monitor/egosuite_progress.json`,
  解析器 `parse_egosuite_progress`。**分母永远是 3621 任务/4053h**, 不再用 p2 的局部分母
- p2 的 TASK_START total=3127 是"无 .p2_done 任务数", 含 1356 个已转 20% 的 standard,
  不代表真实剩余量——以总盘进度条为准

## 6. 剩余全链路(当前执行顺序, 2026-09-14 起由 master_chain.sh OS 级自治)

1. ~~三路时间轴修复~~ ✅ 全部完成(body 615/615, standard 479/479, 补漏 32/32, p2 晚合并 55/55)
2. stts 快扫 ✅(22 受损文件已 POC 重建清零)
3. durcap v2 重轮转 487 任务(跑中) → 0.5.0 抽检 5 任务 → 上传 body 2002 完整任务
   (master_chain.sh 自动接力, 失败写 CHAIN_ABORT; cron 看门狗每小时检查)
   **p2 已于 2026-09-14 提前并行启动**(durcap 的 487 任务 ⊆ body 完整 2002, 与 p2 剩余
   1791 任务文件级不相交; durcap IO 型/p2 CPU 型资源互补; master_chain 第 4 步已改为幂等跳过)
4. p2 每完成一批: stts 快扫 → 漂移扫描 → 修复 → bounds → durcap → 0.5.0 抽检 → 追加上传
5. standard 1356 任务 p2 补完 80% 后同流程处理

## 7. 新批次数据的预防设计(避免重蹈覆辙, 详见 skill: egosuite-convert)

1. 视频轮转从转换期就守 ≤960s 且按 episode 边界切 —— 省掉整条 durcap 工序
2. 写视频用整数 tick 时间基(timescale 取 fps 整数倍), pts/dts 从 0 起严格连续 —— 杜绝三类漂移
3. 禁止盲目 concat -c copy 拼异源 mp4; 任何合并后必过 stts+ctts 快扫和漂移扫描
4. 30fps 网格 + source_timestamp 透传; 透传列全保留; merge 后必跑 fix_subtask_bounds
5. 上传前验证链全绿: stts → 漂移 worst_dev=0 → 时长 ≤960s → bounds → 配对 MAE<2 → 0.5.0 官方加载 PASSED
6. 长任务 setsid 脱离会话 + OS 级自治链 + cron 看门狗; 监控分母用全量总盘(3621 任务/4053h)

## 8. 已知源数据缺陷(转换时丢弃, 完整性判定时需豁免)

- `EgoStandard-body/Foam Protective Wrapping for Fruits and Vegetables` 的
  `6825bf2c-8de7-459a-b98a-53d9dc9ae28a.mcap`: head_right 视频只有 420 包
  (pose 时钟 12934 帧), 源缺陷, staging 按设计丢弃。该任务 9/10 集即为完整。
- 完整性判据(info.json total_episodes >= mcap 数)对这类任务永远不满足,
  后续批次处理时需维护豁免清单(本节)。
