# 按机械臂类型分组

`group_by_robot_type.py` 把一个多子数据集父目录 P 按**机械臂类型**分组，
输出到 Q。**不复制数据**，只创建绝对路径软链接。

## 用法

```bash
python3 group_by_robot_type.py \
  --input /path/to/P \
  --output /path/to/Q
```

只有两个参数：

- `--input`：P，多机械臂数据集的父目录；
- `--output`：Q，输出目录。

## 输入

```text
P/
  Unitree_G1edu_u3_place_towel/
    meta/info.json
    data/...
  Agilex_Cobot_Magic_fold_towel/
    meta/info.json
  ...
```

只处理 P 的**直接子目录**，且必须存在 `meta/info.json`。

## 输出

```text
Q/
  Unitree_G1/
    Unitree_G1edu_u3_place_towel -> /abs/path/P/Unitree_G1edu_u3_place_towel
  Agilex_Cobot_Magic/
    Agilex_Cobot_Magic_fold_towel -> /abs/path/P/Agilex_Cobot_Magic_fold_towel
  unknown/
    ...
  manifest.json
```

- Q 下的子目录名是**规范化的机械臂类型**；
- 优先按数据集前缀判断机械臂类型（例如 `Agilex_Cobot_Magic_`、`Unitree_G1edu_`）；
- 软链接使用**绝对路径**；
- 每个链接指向 P 中对应的子数据集目录；
- 不复制 parquet、视频、meta。

## 分组名从哪里来

优先按数据集名前缀映射：

| 数据集前缀 | 分组 |
|---|---|
| `Agilex_Cobot_Magic_` / `Agilex_Split_Aloha_` | `Agilex_Cobot_Magic` |
| `Unitree_` / `G1edu-` | `Unitree_G1` |
| `Airbot_` / `AIRBOT_` | `Airbot_MMK2` |
| `Galaxea_` / `R1_Lite_` | `Galaxea_R1_Lite` |
| `Leju_Kuavo_` / `leju_robot_` | `Leju_Kuavo_4` |
| `Realman_` / `RMC-AIDA-L_` | `Realman_RMC-AIDA-L` |
| `AgiBot-` / `AgiBot_` | `AgiBot_G1` |
| `AI2_` / `alpha_bot_` | `AI2_AlphaBot_2` |
| `Galbot_` | `Galbot_G1` |
| `Tianqin_` / `Tianqing_` | `Tianqing_A2` |

如果前缀不匹配，再尝试原始 `robot_type`；仍然无效就放到：

```text
Q/unknown/
```

`manifest.json` 会记录每个数据集的 `robot_type_raw`、`robot_type_source`
和 `group_rule`，方便你追溯它为什么被分到该组。

## 注意

原始 `robot_type` 可能不一致。例如：

```text
Agilex_Cobot_Magic_close_drawer -> robot_type = Agilex_Cobot_Magic
Agilex_Cobot_Magic_fold_towel  -> robot_type = aloha
```

按机械臂类型分组时，它们都会进入 `Agilex_Cobot_Magic/`。

## 重复运行

Q 里会写 `manifest.json`。再次运行时：

- 如果 Q 是上次本脚本生成的，并且 `source` 相同，会先安全删除上次的符号链接和空目录，再重建；
- 如果 Q 非空但不是本脚本生成的，会报错，不会删除；
- 如果 Q 里出现非 symlink 的真实文件/目录，会报错，不会覆盖。

如果上次运行中断且没有生成 `manifest.json`，需要手动清理 Q 后重跑。
