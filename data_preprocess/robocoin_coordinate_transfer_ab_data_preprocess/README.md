# RoboCOIN A1/B1 坐标对齐消融数据构建

本目录用于在另一台服务器上，从已经由
[`petrichorz/data-cleaning`](https://github.com/petrichorz/data-cleaning)
处理好的全量 RoboCOIN 55D 数据集 A，生成两个**逐 episode、逐帧对齐**的数据集：

| 输出 | 含义 |
|---|---|
| **A1** | A 的子集；EEF 14:28 保持 A 的 `eef_sim_pose`（已对齐到 base/脚底中心） |
| **B1** | 与 A1 相同的机器人、episode、帧、视频；EEF 14:28 换成各机器人 **URDF root frame 下的 FK** |

A1 与 B1 的差异**只允许**出现在：

```text
state_unified[:, 14:28]
action_unified[:, 14:28]
mask_state[:, 14:28]
mask_action[:, 14:28]
reliability[:, 14:28]        # 若 A 有该列
```

其余 55D 槽位、原始列、索引、时间戳、视频全部一致。

---

## 1. 固定规则

| 项 | 规则 |
|---|---|
| 资格 | `strict = Tier 1 + Tier 2` |
| 排除 | Galbot G1、Tianqing A2、Galaxea 7DoF、AI2 AlphaBot 2 |
| A1 范围 | 与 B1 完全相同的 episode 集合 |
| episode 编号 | 重新从 0 开始编号；manifest 保留 old id |
| 视频 | A1/B1 都软链接到原始 A；不复制 |
| A1 形式 | 物理数据集（重写 parquet） |
| B2 | 不做 |
| EEF 构造 | URDF-root FK；**不乘 `R,t`** |
| 关节源 | A 的 `state_unified/action_unified[:, 0:14]` |
| 额外关节 | AgiBot 的 `body_joint1/2` 从原始 `source.*` 列取 |
| 统计量 | `stats.json` 继承 A，不重算 |
| 并行 | 默认 `--jobs 4`，按子数据集多进程；`--jobs 1` 为串行 |

预计严格模式覆盖约 **614 个仓库 / 95.19M 帧 / 67.7% 帧**。
具体以实际扫描报告为准。

---

## 2. 快速开始

```bash
cd robocoin_coordinate_transfer_ab_data_preprocess

# 依赖：numpy + pyarrow
python3 -m pip install -r requirements.txt

# 先看资格和规模，不写 A1/B1 数据
python3 make_a1_b1.py \
  --input /path/to/A \
  --output /path/to/ablation \
  --dry-run

# 正式生成（默认按子数据集 4 进程并行）
python3 make_a1_b1.py \
  --input /path/to/A \
  --output /path/to/ablation

# 单进程（便于逐数据集看日志）
python3 make_a1_b1.py \
  --input /path/to/A \
  --output /path/to/ablation \
  --jobs 1
```

`--input` 支持两种形式：

- 一个包含多个子数据集的根目录：每个直接子目录都有 `meta/info.json`；
- 单个 LeRobot v3.0 数据集目录。

URDF/calibration 默认从**脚本所在目录**解析：

```text
<脚本目录>/urdf
<脚本目录>/urdf/validation/calibration.json
```

也可以显式指定：

```bash
python3 make_a1_b1.py \
  --input /path/to/A \
  --output /path/to/ablation \
  --urdf-root /path/to/urdf \
  --calibration /path/to/urdf/validation/calibration.json
```

其他可选参数：

```text
--jobs N      默认 4；按子数据集并行，N=1 时串行
--dry-run     只扫描、只写 reports，不写 A1/B1 数据
--overwrite   输出目录非空时，替换 A1/B1/reports/manifest.json/run.log
```

---

## 3. 输出结构

```text
<output>/
  A1/
    <subdataset_name>/
      data/
      meta/
      videos -> <A>/<subdataset_name>/videos   # 相对软链接
      qc/
  B1/
    <subdataset_name>/
      data/
      meta/
      videos -> <A>/<subdataset_name>/videos   # 相对软链接
      qc/
  reports/
    eligibility.json
    fk_quality.json
    diff_summary.md
    provenance.json
  manifest.json
  run.log
```

A1/B1 的每个子数据集都是一个独立的 LeRobot v3.0 数据集，可以单独加载。

---

## 4. 文件说明

| 文件 | 用途 |
|---|---|
| `make_a1_b1.py` | 主脚本：扫描、生成 A1/B1、校验、写报告 |
| `ab_fk.py` | 最小 URDF 解析 + 向量化 FK + 四元数转换 |
| `ab_progress.py` | 进度条、处理速率、剩余时间 |
| `urdf/models/` | 本包需要的推荐 URDF |
| `urdf/validation/calibration.json` | 家族级 URDF/EEF link/关节映射/残差/`R,t` |
| `urdf/validation/manifest.json` | URDF 清单（参考） |
| `docs/PRINCIPLE.md` | 原理：55D、A1/B1、FK、calibration、Tier |
| `docs/USAGE.md` | 另一台服务器上的逐步操作命令 |
| `docs/CALIBRATION.md` | `calibration.json` 字段和 caveat |
| `docs/URDF_SOURCES.md` | URDF 来源与授权 |
| `requirements.txt` | Python 依赖（numpy + pyarrow） |
| `NOTICE.md` | 第三方 URDF 说明 |

---

## 5. 重要注意事项

- A1/B1 的 `videos` 是**相对软链接**；移动输出目录时，A 的相对位置也要保持，否则链接会断。
- `stats.json` 直接继承 A，**没有按子集/B1 重算**。如果训练使用它做归一化，A1/B1 会共享同一套统计量。
- Tier 2 的左臂残差较大；`reports/fk_quality.json` 分别给出：
  - `fit_*`：每个子数据集自己拟合 `R,t` 后的残差（更接近模型误差）；
  - `cal_*`：使用 `calibration.json` 固定 `R,t` 的残差（如果 A 的 frame 和标定样本不同，会很大）。
- 如果 A1 的 EEF 四元数 mask 为 0（部分旧数据只填位置），B1 仍会填四元数；姿态对比会显示 `rot_count=0`。
- 本脚本只支持 LeRobot **v3.0** 的 `meta/episodes/*.parquet` + `data/chunk-*/file-*.parquet` 结构。

更详细的原理见 [`docs/PRINCIPLE.md`](docs/PRINCIPLE.md)，
逐步操作见 [`docs/USAGE.md`](docs/USAGE.md)。
