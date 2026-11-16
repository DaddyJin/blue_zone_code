# 另一台服务器上的操作步骤

以下命令都假设你已经把本目录同步到目标服务器：

```text
robocoin_coordinate_transfer_ab_data_preprocess/
```

并把全量 A 放在目标服务器上的某个目录，例如：

```text
/data/robocoin/A
```

---

## 0. 同步与环境检查

### 同步到目标服务器

在本仓根目录把本目录加入 git（只提交代码、URDF、calibration 和文档，不提交 A/输出）：

```bash
git add data_preprocess/robocoin_coordinate_transfer_ab_data_preprocess
git commit -m "add RoboCOIN A1/B1 coordinate-transfer builder"
git push
```

目标服务器：

```bash
git pull
cd data_preprocess/robocoin_coordinate_transfer_ab_data_preprocess
```

本包只有约 600KB（URDF 只保留 FK 需要的 `.urdf`，没有 mesh / sources），
适合直接通过 git 传输。

### 环境检查

```bash
cd /path/to/robocoin_coordinate_transfer_ab_data_preprocess

python3 --version
python3 -c "import numpy, pyarrow; print('numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)"
```

如果缺少依赖：

```bash
python3 -m pip install -r requirements.txt
```

本包不依赖 `scipy`、`torch`、`lerobot`。FK 和四元数转换都用 numpy 实现。

---

## 1. 检查 A 的目录结构

A 应该满足以下两种形式之一：

### 形式 1：多个子数据集

```text
A/
  Unitree_G1edu_u3_xxx/
    meta/info.json
    meta/episodes/...
    data/chunk-000/file-000.parquet
    videos/...
  Agilex_Cobot_Magic_xxx/
    ...
```

脚本会扫描 A 的**直接子目录**，只处理带 `meta/info.json` 的目录。

### 形式 2：单个子数据集

```text
A/
  meta/info.json
  meta/episodes/...
  data/...
  videos/...
```

检查：

```bash
find /data/robocoin/A -maxdepth 3 -name info.json | head
```

脚本当前只支持 LeRobot **v3.0**：

```bash
python3 - <<'PY'
import json
p='/data/robocoin/A'
# 如果是多子数据集，改成某个子数据集的 meta/info.json
print(json.load(open(p+'/meta/info.json'))['codebase_version'])
PY
```

如果是 `v2.1`，需要先确认 `data-cleaning` 的输出版本；本脚本不处理 v2.1 的
`episode_XXXXXX.parquet` 布局。

---

## 2. 先跑 dry-run

dry-run 只扫描、做一次样本 FK 校验，并写报告，不生成 A1/B1 数据：

```bash
python3 make_a1_b1.py \
  --input /data/robocoin/A \
  --output /data/robocoin/ablation \
  --dry-run
```

默认 `--urdf-root` 和 `--calibration` 相对脚本目录：

```text
<脚本目录>/urdf
<脚本目录>/urdf/validation/calibration.json
```

如果 URDF/calibration 放在别处：

```bash
python3 make_a1_b1.py \
  --input /data/robocoin/A \
  --output /data/robocoin/ablation \
  --urdf-root /data/robocoin/urdf \
  --calibration /data/robocoin/urdf/validation/calibration.json \
  --dry-run
```

查看结果：

```bash
cat /data/robocoin/ablation/reports/diff_summary.md
python3 -m json.tool /data/robocoin/ablation/reports/eligibility.json | less
```

dry-run 的 `included` 表示该子数据集有匹配的 URDF/映射；正式跑之前建议先确认
`excluded` 里的原因符合预期。dry-run 只会写 `reports/manifest.json/run.log`，
不会写 A1/B1；确认后可以直接跑正式命令，不需要先删目录。

---

## 3. 正式生成 A1/B1

```bash
python3 make_a1_b1.py \
  --input /data/robocoin/A \
  --output /data/robocoin/ablation
```

终端会显示类似：

```text
[##############--------------]  50.64% Build Unitree_... 278/549 frames
9.8k frames/s | elapsed 00:10:12 | ETA 00:10:12 | ... ep 0
```

每个阶段：

1. 扫描子数据集、识别 family/Tier、检查 URDF/DoF；
2. 逐 episode 读 A、算 B1、校验 A1/B1；
3. 同时写 A1/B1 parquet；
4. 建视频软链接；
5. 写 meta、reports。

如果某个 episode 的 FK/结构校验失败，该 episode 会同时从 A1 和 B1 中剔除，
原因写入：

```text
<output>/reports/eligibility.json
<output>/reports/fk_quality.json
<output>/manifest.json
<output>/run.log
```

---

## 4. 检查输出

### 4.1 目录

```bash
find /data/robocoin/ablation -maxdepth 2 -type d | sort | head -40
```

应该看到：

```text
ablation/A1/<subdataset>/
ablation/B1/<subdataset>/
ablation/reports/
```

### 4.2 视频软链接

```bash
ls -l /data/robocoin/ablation/A1/<subdataset>/videos
ls -l /data/robocoin/ablation/B1/<subdataset>/videos
```

应该是 symlink，指向 A 的 `videos/`。A1/B1 不复制视频。

### 4.3 episode 数、帧数

```bash
python3 - <<'PY'
import json, glob, os
for cond in ('A1','B1'):
    print('==', cond)
    for p in sorted(glob.glob(f'/data/robocoin/ablation/{cond}/*/meta/info.json')):
        d=json.load(open(p))
        print(os.path.basename(os.path.dirname(os.path.dirname(p))), d['total_episodes'], d['total_frames'])
PY
```

A1/B1 的同名子数据集，`total_episodes` 和 `total_frames` 必须相同。

### 4.4 非 EEF 槽位一致

```bash
python3 - <<'PY'
import glob, json, os, numpy as np, pyarrow.parquet as pq
root='/data/robocoin/ablation'
for ds in sorted(os.listdir(root+'/A1')):
    a=glob.glob(f'{root}/A1/{ds}/data/**/*.parquet', recursive=True)
    b=glob.glob(f'{root}/B1/{ds}/data/**/*.parquet', recursive=True)
    # 数据文件边界一致时，逐文件比较
    assert len(a)==len(b), ds
    for fa, fb in zip(sorted(a), sorted(b)):
        ta=pq.read_table(fa); tb=pq.read_table(fb)
        assert ta.num_rows==tb.num_rows
        for kind in ('state_unified','action_unified'):
            x=np.asarray(ta[kind].to_pylist()); y=np.asarray(tb[kind].to_pylist())
            assert np.array_equal(x[:,:14], y[:,:14]), (ds,kind,'arms')
            assert np.array_equal(x[:,28:], y[:,28:]), (ds,kind,'tail')
        for kind in ('mask_state','mask_action'):
            x=np.asarray(ta[kind].to_pylist()); y=np.asarray(tb[kind].to_pylist())
            assert np.array_equal(x[:,:14], y[:,:14]), (ds,kind,'arms')
            assert np.array_equal(x[:,28:], y[:,28:]), (ds,kind,'tail')
    print('OK', ds)
PY
```

### 4.5 残差

```bash
python3 -m json.tool /data/robocoin/ablation/reports/fk_quality.json | less
```

看两组数：

- `fit_pos_rms_mm`：当前子数据集自己拟合 `R,t` 后的位置 RMS；
- `cal_pos_rms_mm`：用 `calibration.json` 固定 `R,t` 后的位置 RMS。

如果 `fit` 很小、`cal` 很大，说明该子数据集的 A frame 和标定样本不同，
但 URDF/关节映射本身仍然能对齐。

---

## 5. 加载

A1/B1 的每个子数据集都是独立的 LeRobot v3.0 数据集。用你现有的 LeRobot loader
分别指向：

```text
/data/robocoin/ablation/A1/<subdataset>
/data/robocoin/ablation/B1/<subdataset>
```

例如（具体 API 以你使用的 LeRobot 版本为准）：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="local/A1/Unitree_xxx",
    root="/data/robocoin/ablation/A1/Unitree_xxx",
)
```

如果 loader 需要 `task`/`split` 参数，按你当前训练代码的约定传入。

---

## 6. 重跑与中断

脚本没有实现断点续跑。若中途中断：

```bash
python3 make_a1_b1.py \
  --input /data/robocoin/A \
  --output /data/robocoin/ablation \
  --overwrite
```

`--overwrite` 会删除并重建：

```text
<output>/A1
<output>/B1
<output>/reports
<output>/manifest.json
<output>/run.log
```

不会动 A。

---

## 7. 常见问题

| 现象 | 原因/处理 |
|---|---|
| `no dataset found` | `--input` 下没有 `*/meta/info.json`；检查是否指向了 A 的父目录 |
| `only LeRobot v3.0 ... supported` | A 是 v2.1；需要先转换或调整脚本 |
| `KeyError: data/file_from_offset` | 旧脚本假设 `meta/episodes` 一定有该字段；当前脚本已支持缺失 offset（按 episode_index 过滤或一 episode 一文件），更新代码即可 |
| `extra_joint_missing: ... raw width ...` | 子数据集的 raw 布局和 calibration 不一致，无法提供 AgiBot 的 body/head 关节；该子数据集会被排除（例如 `AgiBot-g1_box_storage_tool` 实际是 17 维的 `ruantong_a2d`） |
| `family_not_in_calibration` | 子数据集前缀无法识别；在 `resolve_family()` 中补充规则 |
| `dof_mismatch` | 数据关节数和 calibration 不一致；检查 6/7-DoF |
| `ai2_public_urdf_mismatch_strict` | AI2 在 strict 下被排除，符合预期 |
| `galaxea_7dof_no_7dof_urdf` | Galaxea 7DoF 没有 7-DoF 模型，符合预期 |
| 视频链接断 | 输出目录被移动，或 A 的相对位置改变；重新建立 symlink |
| 残差 `cal_*` 很大、`fit_*` 很小 | A 的 `eef_sim` frame 和标定样本不同；看 `fit_*` |
| `rot_count=0` | A1 的 EEF 四元数 mask 为 0（旧数据只填位置）；B1 仍会填四元数 |

---

## 8. 传回/归档

建议把以下内容一起归档：

```text
ablation/A1/
ablation/B1/
ablation/reports/
ablation/manifest.json
ablation/run.log
```

视频是软链接，不占额外空间；如果要把数据集搬到另一台机器，需要同时保留 A 的
`videos/`，或把软链接改成实际文件。

---

## 9. 最小命令清单

```bash
cd robocoin_coordinate_transfer_ab_data_preprocess
python3 -m pip install -r requirements.txt

# 1) 快速检查
python3 make_a1_b1.py --input /data/robocoin/A --output /data/robocoin/ablation --dry-run

# 2) 正式生成
python3 make_a1_b1.py --input /data/robocoin/A --output /data/robocoin/ablation

# 3) 看报告
cat /data/robocoin/ablation/reports/diff_summary.md
python3 -m json.tool /data/robocoin/ablation/reports/fk_quality.json | less
```
