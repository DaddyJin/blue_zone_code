# RoboCOIN：独立 LeRobot 0.5.0 加载验收

此包与转换代码分开；不包含数据、环境或任何下载逻辑。使用已安装/离线恢复好的 **LeRobot 0.5.0** Python 环境。29 上实测环境为 Linux ARM64 / Python 3.13；转换环境则是独立的 Python 3.10。

```bash
cd /path/to/robocoin_loader_tests
LEROBOT_PY=/path/to/lerobot_environment/bin/python
OUTPUT=/data/robocoin_unified55_v4

# 必须输出 0.5.0；此命令不会联网。
"$LEROBOT_PY" -c 'import importlib.metadata as m; print(m.version("lerobot"))'

# 逐个输出单元实际加载。建议报告写在 OUTPUT 外，保持输出清单不变。
python3 scripts/official_load.py --output "$OUTPUT" \
  --python "$LEROBOT_PY" --report /data/robocoin_loader_report.json

# 只验证一个单元：
"$LEROBOT_PY" scripts/loader_probe.py /data/robocoin_unified55_v4/one_dataset_unit
```

测试会设置离线模式，使用 CPU，读取每一集的首尾和相邻边界帧，以及全局首/中/尾。逐项比较 55D、mask，并将官方加载结果与精确 PTS 参考读取结果进行 RGB 逐像素比较。报告记录实际版本、读取索引和图像数量。退出码 0 且报告 `passed=true` 才通过；单元失败不会阻止其他单元验收。

加载方式是官方 `LeRobotDataset` 的薄兼容子类，代码在 `vendor/standard_format/tools/lerobot_compat.py`：

1. 没有子任务标注时，将标准格式的 subtask=-1 读成空文本；
2. 按 parquet 的真实 Arrow 类型加载原始列，保留 RoboCOIN 的长度为 1 的 list，避免官方 feature 推断把它误读成 scalar。

不会改写数据、已安装的 LeRobot 库或视频寻找算法。训练代码也应通过此包提供的 `load_dataset(root)` 加载；不能把“兼容类测试通过”写成“未经适配的原生类直接通过”。

```python
import sys
sys.path.insert(0, "/path/to/robocoin_loader_tests/vendor/standard_format/tools")
from lerobot_compat import load_dataset

dataset = load_dataset("/data/robocoin_unified55_v4/one_dataset_unit")
sample = dataset[0]
```

此项是实际训练加载路径验收，不替代转换包的全量行、视频 packet 和 SHA256 核对。样例验收报告附在 `reports/official_load_final.json`。
