# 54 个仓库：维度问题与修复结果

> 本页记录 v2 的维度修复；v3 保留该能力，运行版本及输出目录以主 README 为准。

## 结论

此前把这 54 个仓库因 `shape` 与 `names` 数量不符直接跳过，过于保守。实际抽查表明：**Parquet 长度与 shape 一致，问题是 names 不完整，不是已证实的数值维度损坏。**

| 类别 | 仓库数 | 原始元数据 | 实际 Parquet | 未命名通道的样例情况 |
|---|---:|---|---|---|
| G1edu-u3 | 35 | state 和 action 各声明 30 维，分别只列 28 个名称 | 两列均为 30 维 | 第 28、29 列（从 0 开始）有变化，不可删除或视为填充 |
| R1 Lite | 19 | state 为 16 维/16 名；action 为 18 维/16 名 | state=16，action=18 | action 第 16、17 列在抽查中全为 0；不据此认定全量都是填充 |

核对范围：从 ModelScope 的 RoboCOIN 同名公开仓库，分别读取首集和末集 Parquet，共 **108 文件、34,418 行**。所有样例长度一致，未发现空向量或非有限数值。54 份元数据的 features 与已保存 HF 快照一致；两个已下载 HF 代表库的首集 Parquet 与 ModelScope 文件 SHA256 一致。

这是对所有 54 个仓库的首尾抽样，不是对其全部 episode 的扫描。各仓库名单、字段数量、样例统计见 `reports/dimension_audit.json` 和 `research/dimension_repairs.json`。

## 修复方式

1. 只对这 54 个仓库、两个精确匹配的已审查 schema 生效，不对任意缺名数据自动补齐。
2. 不修改输入文件，不改向量宽度，不删值、不补数值。原始列仍完整保存为 `source.*`。
3. 内部有效元数据补上 `unnamed_source_component_28` 等**中性名称**，明确语义未知。原始元数据另行原样保存。
4. 继续使用已发布的顺序映射明确字段：G1 的双臂、R1 Lite 的双臂和夹爪等；未命名两维不填入任何有物理定义的 55D 槽。
5. G1 的每手 7 个关节超过现有 6 维手部槽，仍保留完整原始列，不随意截断。这与本次缺名问题是两件事。
6. 正式转换先扫描所有本地 Parquet 的相关列、所有行；只有每行长度符合声明才启用此修复。真正长度不齐或空向量仍记录异常，其他仓库继续。

输出增加：

- `meta/metadata_repair.json`：原名称、未知列索引、规则、所有输入文件的扫描记录。
- `meta/source_info_effective.json`：名称补齐后的内部元数据。
- `meta/source_metadata/`：保持未修改的原始元数据。

不能把中性名称当作找到了真实物理含义。作者的 [G1 控制器](https://github.com/FlagOpen/RoboCOIN/blob/b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd/src/lerobot/robots/g1_29_robot/g1_29_robot.py#L156) 明确了双臂/双手 28 维的顺序，但没有说明这批发布数据额外两维的来源，因此本次没有猜测其为头部、夹爪或其他量。

## 已完成验证

- 54 个仓库的 108 个 Parquet：长度扫描、55D 数值映射和未知通道不被使用，全部通过。
- 两个原先跳过的代表库恢复转换；与其他代表库一起，共 29 个仓库、58 集通过端到端转换及原列/视频保真核对。
- 另行实际执行 LeRobot 0.5.0 加载与边界图像比较，结果见 `reports/official_load_v2.json`。
- 故意制造一行 29 维、其他行 30 维的反例，确认不会被名称修复放行；不在名单中的仓库也不会误套规则。

## 已有 v0 结果如何补齐

使用新的 v2 工具包。保持原结果不变，仅将这 54 个仓库写到新输出目录，再把两批输出加入训练清单；不需要重做原已完成仓库。

```bash
.venv/bin/python scripts/convert_repos.py convert \
  --input /data/robocoin_extracted \
  --output /data/robocoin_recovered54_v2 \
  --work /data/robocoin_recovered54_work \
  --only-repairable \
  --workers 4 --video-workers 4 --decode-threads 8 --cpu-budget 128

.venv/bin/python scripts/verify.py \
  --plan /data/robocoin_recovered54_work/plan.json \
  --output /data/robocoin_recovered54_v2 \
  --report /data/robocoin_recovered54_verification.json
```

`--only-repairable` 将预期范围限定为这 54 个仓库；仍应检查 `skipped`、`missing_expected_repos` 和每单元的修复记录。全新处理全部数据时，不加该选项即可。不要把新版直接覆盖到正在运行的旧任务上，也不要用新版覆盖旧输出目录。
