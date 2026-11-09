> **历史流程（v1）**：本文保留已有 EgoSuite 的处理约定；固定尺寸、30fps、时长及特定目录要求不再是通用约束。新数据集请从 [Unified55 v2 主规范](../README.md) 和 [通用工具](../tools/README.md) 开始。

# scripts/ 用法速查

所有脚本的权威规范见 `../README.md`。运行前确认环境:
- **forge_venv**(`/home/z00827204/forge_venv/bin/python`):av + pyarrow,跑视频/parquet 类脚本
- **finevla**(`/home/z00827204/miniconda3/envs/finevla/bin/python`):obs sdk + pandas,跑上传/转换
- **lerobot050**(`/home/z00827204/miniconda3/envs/lerobot050/bin/python`):官方加载校验,
  必须 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 LD_LIBRARY_PATH=/home/z00827204/miniconda3/envs/lerobot/lib`

## 转换类

### convert_agibot_subtasks.py(forge_venv)
agibot 式 lerobot v3(episodes.jsonl 里有 action_config)→ 050 三件套:
逐帧 `subtask_index` 列 + `meta/subtasks.parquet` + info.json 注册。幂等(`.subtask050` 标记)。
```
python convert_agibot_subtasks.py <agibot_root> --workers 8 [--tasks task_352,task_354]
# agibot_root = 直接含 task_* 的那层; 每个 task 下有 canonical_55d/
```

### egostandard_staging.py(forge_venv)—— 参考 adapter
EgoSuite mcap → 临时 lerobot v3.0 staging。**新数据集照此写自己的 staging**:
视频 960×728@30fps、info.json/features/词表/episodes 全量、透传列保留、action_config 写 episodes.jsonl。
```
python egostandard_staging.py --tasks <subset>/<task> [--sample-every 1] --out <staging_dir>
```

### egostandard_remerge_videos.py(forge_venv)
逐集 mp4 → ~500MB 轮转会并(concat -c copy),重写 episodes 索引,ffprobe 校验帧数。
```
python egostandard_remerge_videos.py <out_root> --workers 3
```

### egostandard_durcap_rotate2.py(forge_venv)
>960s 视频按 episode 边界重切(lerobot 0.5.0 float32 硬伤,§5)。≤960s 文件硬链接零成本。幂等(`.durcap_done`)。
```
python egostandard_durcap_rotate2.py <task_dir>
```

## 校验类

### egostandard_050_official_load.py(lerobot050)—— 最权威判据
lerobot 0.5.0 `LeRobotDataset` 官方加载 + `meta.subtasks` + `__getitem__` subtask 文本
+ 随机 500 帧(seed=0)文本↔索引一致(触发真实视频解码)。**任何本格式数据集通用**。
```
LD_LIBRARY_PATH=.../envs/lerobot/lib HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python egostandard_050_official_load.py <task_dir>   # 输出 ALL CHECKS PASSED 即过
```

### validate_agibot_subtasks.py(forge_venv)
agibot 050 转换的逐帧全量校验(action_config 重建期望数组 vs 实际逐元素比对)+ `--official` 官方加载。
```
python validate_agibot_subtasks.py <agibot_root> [--tasks ...] [--official]
```

### egostandard_timeline_drift_scan.py(forge_venv)
drift 检测:每相机 |Σ容器时长 − 总帧数/30| ≤ 0.5s。全量扫描或单任务。
```
python egostandard_timeline_drift_scan.py <root_or_task_dir>
```

### stts_census.py(forge_venv)
mp4 stts 盒健康普查(只读 moov 几 KB,毫秒级/文件)。输入为任务清单 json。
```
python stts_census.py <task_list.json> --workers 8
```

## 修复类

### egostandard_fix_subtask_bounds.py(forge_venv)
以 episodes.jsonl 的 action_config 为准,幂等重写逐帧 subtask 边界。词表只增不改。
```
python egostandard_fix_subtask_bounds.py <root> --workers 4 [--tasks ...]
```

### mp4_timescale_fix.py / mp4_retime_mixed.py(forge_venv)
drift 修复两级:
- `uniform` 型:`mp4_timescale_fix.py <file.mp4> --write`(mdhd/elst 补丁,无损)
- `mixed` / `garbage` 型:`mp4_retime_mixed.py <src> <dst> --mode pts|poc`
  (pts=按容器呈现序;poc=H.264 图像序重建,用于 ctts 离网格的重污染文件)

## 上传类

### obs_marker.py(finevla)
所有数据文件传完后,生成并**最后**上传 `_UPLOAD_COMPLETE.json`(下游只认这个标记)。
```
python obs_marker.py <task_dir> <bucket> <prefix>
# 凭证读 /home/z00827204/data_pipeline/.env
```
