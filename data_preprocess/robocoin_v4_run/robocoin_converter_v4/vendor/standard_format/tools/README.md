# 转换工具

这些工具只读源数据，在**新的输出目录**生成 Unified55 v2。不会修改或覆盖已有数据集。先读 [主规范](../README.md)，再选一个入口。

## 1 选择入口

| 原始格式 | 操作 |
|---|---|
| 已有 LeRobot v2.1／v3 | 填写语义 profile，`from_lerobot.py` 生成视频与数据行引用 |
| 一集一个视频，无动作 | `make_video_manifest.py` 直接生成配置 |
| 每集已有 Parquet + 视频 | 直接填写 manifest，参考 `examples/robot.json` |
| HDF5／RLDS／自定义容器等 | 按 `adapter_template.py` 提取逐样本列和视频引用；不由通用工具猜测语义 |
| 小型 NPZ 数值／字符串数组 | 可先用 `npz_to_parquet.py` 转为每集 Parquet |

```bash
python -m pip install -r tools/requirements.txt

# 已有 LeRobot：从示例复制 profile，删除 episodes 并修改真实字段；工具补回 episodes
python tools/from_lerobot.py /data/source profile.json manifest.json

# 纯视频：一文件一集；以下要求本单元各文件原本均为 25fps 且尺寸一致
python tools/make_video_manifest.py /data/clips/*.mp4 --dataset-id example \
  --policy native --fps 25 --output manifest.json

# 混合规格：显式指定输出尺寸、FPS，保留帧数并重定时
python tools/make_video_manifest.py /data/clips/*.mp4 --dataset-id example \
  --policy normalize --fps 30 --size 960 728 --output manifest.json

python tools/convert.py manifest.json /data/new_output
python tools/validate.py /data/new_output --decode --samples 8 --output validation.json
python tools/smoke_load.py /data/new_output --samples 8
```

`probe_videos.py DIR --output inventory.json` 只检查头信息，不能代替完整解码质检。`from_lerobot.py` 保留源划分；存在多个 split 时在 profile 显式选一个 `split`，不会自动把测试集当训练集。源任务词表损坏会报错，须用原始语言字段证明并修复。

## 2 Manifest 需要填什么

[JSON Schema](manifest.schema.json) 可供编辑器／agent 检查结构，也可运行 `python tools/check_manifest.py manifest.json`；转换前会自动执行。物理语义仍须人工核实。路径相对于 manifest 所在目录，也可用绝对路径。

| 字段 | 必填？ | 内容 |
|---|---|---|
| `dataset_id`、`profile_id` | 是 | 交付单元 ID；语义 profile ID，不同含义使用不同 profile |
| `data_kind`、`robot_type` | 是 | robot／ego／observation_only／video_only；真实机器人或主体名，纯视频可用 none |
| `source` | 是 | `id`、版本或 `revision`，以及必要的来源证据／说明 |
| `fps` | 是 | 本单元输出标称 FPS；normalize 时与 media.fps 相等 |
| `media` | 是 | `{ "policy":"native" }` 或 `{ "policy":"normalize", "fps":30, "size":[960,728] }` |
| `state_fields` | 是 | 55D 映射列表；没有状态填 `[]` |
| `action` | 是 | `mode`；source 另需 `fields`、`terminal: keep/mask`；none 无映射 |
| `frames` | 有坐标量时 | 参考系字典，记录 kind、原点、轴方向、证据／未知状态 |
| `time` | 建议显式填 | `kind`、源时钟 `column` 和 `unit: s/ms/us/ns`；无时钟用 unknown；nominal 需 fps |
| `episodes` | 是 | 非空 episode 列表，见下表 |
| `passthrough_columns` | 可选 | 原列名列表或 `"all"`；保留为 `source.*`，默认不透传 |
| `task_column` | 按来源 | 逐帧 str 列；否则用 episode.task；LeRobot 接入工具会生成 ID／词表映射 |
| `invalid_sample_column` | 可选 | bool 列，true 时屏蔽该行 state/action；不删除原帧 |
| `calibration_columns` | 有校准且改尺寸时 | 例如 `{"main":{"K":"camera_K","P":"camera_P"}}`，3×3／3×4 展平列；工具按图像变换更新 |
| `parquet_rows` | 可选 | 目标数据分片行数，默认 50000；不拆开一个 episode |

`media.max_file_bytes` 可设置无损合并的目标文件大小，默认约 500MB；单个不可拆的源共享文件可超过该值。通用转换器按集处理、限制文件缓存，未提供不受控的全库并行调度；大规模转换建议先做小样本实测，再在交付单元之间限额并行。

工具在 sidecar 登记原始列名／类型和数据引用。未透传的原始文件必须继续可访问；若要交付独立副本，应复制这些原始文件或补充可长期访问的版本化地址，不能只留下即将删除的临时路径。ragged／struct 等复杂列先单独保存，通用 LeRobot feature 不直接接收它们。

### 每个 episode

| 字段 | 要求 |
|---|---|
| `source_episode_id` | 建议保留可回溯原集 ID；省略时用输入顺序 |
| `videos` | 相机名 → 视频引用；所有集相机集合一致；无视频用 `{}` |
| `data` | 有状态／动作／真实时钟时需要；每集行数须等于视频帧数；纯视频可省略 |
| `length` | 无视频时必须显式给出正整数 |
| `fps` | 源集标称 FPS，未给则由视频头／单元配置取得；VFR 不等于真实固定采样率 |
| `task` | 没有逐帧任务列时使用；未知填空字符串，不编造指令 |
| `split`、`split_group_uid` | 建议显式保留原划分和演示分组，防止相同演示泄漏到训练与测试 |
| `time` | 允许覆盖单元 time，适用于不同时钟来源 |
| `segments` | 可选，`action_text, skill, start_frame, end_frame`；end 不含；按起点排序、不重叠 |
| `excluded` | 可选，整个 episode 屏蔽监督，保留数据；**不会容忍无法解码的坏码流** |

数据引用支持：

```json
{
  "data": [
    {"path":"data/file-0.parquet", "offset":100, "length":20},
    {"path":"data/file-1.parquet", "offset":0, "length":30}
  ],
  "videos": {
    "main": {"path":"shared.mp4", "from_frame":100, "length":50}
  }
}
```

数据也可用简单路径或 `{"path":"file.parquet","episode_index":7}`。视频用 `from_frame` 指定解码顺序中的帧起点。

已有 v3 视频还支持 `{path, from_timestamp, to_timestamp, length, fps}`；工具根据实际 packet PTS 解析区间，再完整解码核实。不能只用 `round(timestamp*fps)` 猜区间。

### 每条 55D 映射

最少填写 `source, slot, owner, quantity, unit, provenance`；owner 必须与本单元 robot_type 一致。`slot` 名称见 `common.SLOTS`，可用 `indices` 选源分量，用 `components` 指定槽内位置。例如只映射左腕 xyz：`slot=left_eef, indices=[0,1,2], components=[0,1,2]`。

| 情况 | 额外填写 |
|---|---|
| 关节／手部／头部 | 按实际映射维数填写有序 `component_names`；单位 rad／rad/s，可用 source_unit 从 deg／deg/s 转换 |
| 末端／腕部 | `frame`、`reference_point`；含旋转时填 quaternion_order；输出 m、xyzw，可从 mm／wxyz 转换 |
| 夹爪 | `gripper.mode=open_01/closed_01/width/unknown`；width 需原始单位 m/mm 和 closed/open 标定端点；输出单位 `"1"` |
| 腰／底盘 | `component_names` 和逐分量 SI `unit` 列表；非 SI 在 adapter 中转换 |
| 派生量 | `provenance=derived`、`method`；假定或未知用 assumed／unknown，工具会屏蔽 |
| 已有逐分量质检 | `valid_column`：每行 bool 标量或与本映射宽度相同的 bool 向量；false 只屏蔽本映射相应分量 |

例如 OXE 重放误差超过 1cm 的 EE 可以给 EE 映射配独立 valid_column，保留该帧有效关节／夹爪。不同字段的质量不能共用一个无依据的总分。

`action.mode=source` 不自动猜时间跨度。若有实测 horizon，填 `dt_column`（秒）与 `dt_kind=measured`；否则 action_dt 保持无效。增量、速度和绝对位姿必须使用可区分的 quantity/profile，训练方据此选用。

## 3 验收与读取

```bash
python tools/selftest.py                  # 仅合成数据，自动清理
python tools/smoke_load.py OUTPUT          # 精确 PTS 参考读取器，numpy / HWC uint8
python tools/smoke_load.py OUTPUT --official # LeRobot + 随附缺失子任务适配器，PyAV 后端
python tools/smoke_load.py OUTPUT --official-raw # 不使用适配器，检验原版兼容性
```

`read_frame.StandardDataset(root, episode_indices=[...])` 提供按行索引、55D、来源 profile／机器人／分组、实际图像和每相机 float64 `.pts`。数据和解码器有缓存上限，兼容进程间重新初始化；训练时可在外层包装成 PyTorch Dataset。不要直接跨 profile 计算一个均值或忽略 mask。

LeRobot 0.5 原版对缺失子任务 -1 没有保护，空词表会报错，非空时可能错取最后一项。`lerobot_compat.load_dataset` 仅在该实例中接管子任务文本读取，缺失时返回空文本与 mask_subtask=false；不修改安装包或数据。该适配器不解决旧视频后端的长片／VFR 查询问题，精确时序使用 StandardDataset。已测版本与范围见 [验证记录](../docs/VALIDATION.md)。通用转换器遇到坏帧、对齐不明、异构列或词表错误会失败；排查隐藏的 `.OUTPUT.building-*`，修正 adapter 后换新目录重试，不能把半成品当交付。

坐标系变换可用 `pose.transform_absolute_pose`，必须提供已知刚体矩阵；没有矩阵时保留原系并登记，参见 [决策说明](../docs/DECISIONS.md)。
