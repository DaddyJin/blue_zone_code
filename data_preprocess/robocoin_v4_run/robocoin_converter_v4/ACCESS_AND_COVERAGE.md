# RoboCOIN 访问修正与覆盖范围

## 62 个失败的原因

HF 的 `RoboCOIN/Cobot_Magic_<任务>` 对应 ModelScope 的 `RoboCOIN/Agilex_Split_Aloha_<任务>`。使用 HF 名访问 ModelScope 会返回 404、“不存在的数据集”；改名后，62 个仓库的元数据和首个 Parquet 的 `PAR1` 文件头全部读取成功，无需登录。

证据来自 [ModelScope 组织目录](https://modelscope.cn/organization/RoboCOIN) 的公开 API：`https://modelscope.cn/api/v1/datasets?Owner=RoboCOIN&PageNumber=1&PageSize=1000`。目录列出 809 个仓库，改名后与 HF 的 809 个训练仓库一一对应。HF 的 pageAssets 不计入训练仓库。

- 完整映射：`research/repository_aliases.json`。
- 访问探测：`reports/modelscope_access_v2.json`（747 个沿用上一轮成功文件头探测，62 个本轮补查；本轮重查全部元数据）。
- 不是全量下载证明：未读取所有视频、所有 Parquet 或逐文件校验全部远端数据。

## 正式输入要求

1. 每个已解压目录保留任一平台的仓库名，内部包含该版本完整的 meta/data/videos。
2. 每个仓库只选一个平台的一份完整版本。转换器读取实际输入，不套用另一平台的集数或帧数。
3. 同一输入出现 HF / ModelScope 两份同仓库时，两份均列入跳过报告，避免重复训练；保留选定版本后重新运行。
4. 改名通过目录映射识别，不要求修改原始文件。输出记录统一仓库 ID、实际目录名、原始元信息及其哈希。
5. 本包支持快照中所有 809 个仓库的结构。缺文件、真实向量宽度错误、坏视频等仍会报告失败或跳过；不能为了“全部成功”隐瞒坏数据。

## 两个平台的版本差异

两站有 9 对仓库的集数或帧数不同。以下是本次元数据快照，均为“集数 / 帧数”：

| HF 仓库名 | HF | ModelScope |
|---|---:|---:|
| Cobot_Magic_open_the_shoebox | 292 / 177,109 | 299 / 185,740 |
| Cobot_Magic_desktop_organization | 1,288 / 2,064,431 | 1,070 / 2,026,628 |
| Cobot_Magic_vase_storage_flower | 453 / 237,198 | 455 / 239,272 |
| Cobot_Magic_food_packaging | 842 / 1,492,934 | 830 / 1,499,538 |
| Cobot_Magic_make_hamburger | 3,077 / 1,596,406 | 3,044 / 1,591,383 |
| Cobot_Magic_pour_water_a | 891 / 1,283,817 | 896 / 1,302,424 |
| alpha_bot_2_press_the_button_b | 34 / 26,258 | 65 / 50,921 |
| AI2_Alphabot_2_package_fruit | 988 / 384,538 | 510 / 209,718 |
| AI2_Alphabot_2_organize_cosmetics | 407 / 330,781 | 405 / 329,188 |

`alpha_bot_2_press_the_button_b` 的 ModelScope 元信息还少了 HF 中的 `gripper_activity_action` 特征声明。转换器保留实际 Parquet 的全部原列；不根据另一站的元信息凭空补该列。

## 验收边界

| 项目 | 已验证范围 |
|---|---|
| 仓库元数据及首个 Parquet 文件头可访问 | 809 / 809，公开接口 |
| 数值结构支持 | 26 种，覆盖 809 / 809 份元数据 |
| 实际转换、原列与 55D 校验、视频内容顺序保真 | 29 仓库、58 集、29,064 行 |
| 缺名修复的实际 Parquet 抽查 | 54 仓库首/尾，共 108 文件、34,418 行（此前完成） |
| LeRobot 0.5.0 实际加载 | 29 个样例输出；每集边界及 RGB 比较，见独立加载包 |
| 全量 809 仓库逐文件完整性 | 未执行，须在正式机转换后验收 |

正式验收同时检查 `verification.json.passed`、`corpus_complete`、`plan.json.skipped`、`missing_expected_repos` 和 `progress.json`。实际输入不完整时，不能只凭已处理部分通过就宣布全量完成。
