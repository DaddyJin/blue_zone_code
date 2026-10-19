# data_preprocess — 团队标准格式数据预处理工具链

本板块收录把各种源数据转换成**团队统一格式**(LeRobot v3.0 官方布局 + 55D 统一动作列)
的**转换/校验代码**。每个子目录是一个独立项目:自带 README、转换脚本、校验脚本。
输入数据均在运行环境就地提供,本仓**只收录代码**(数据/视频/日志已由外层 `.gitignore` 排除)。

团队规范材料(`standard_format/`,含 55D 槽位布局、视频规范、校验链)为内部文档,
不在本仓;各项目 README 中引用它的地方仅为出处说明,不影响运行。

## 三个项目速览

| 项目目录 | 适用数据 | 源格式 | 核心能力 |
|---|---|---|---|
| `gm100-cobotmagic-lerobot-preprocess/` | GM100 cobotmagic(双臂 6-DOF ×2 +夹爪,实机) | lerobot **v2.1**(per-episode) | v2.1→v3.0+55D;Piper URDF **FK 回填 eef**;逐集硬链接;断点续跑 |
| `lingbot-GM-100-preprocess/` | lingbot GM-100(**压缩包交付**,与 GM100 同源数据) | lerobot v2.1(**tar.gz**) | **批量解压**(幂等/原子发布)+ 同上转换 |
| `InternData-A1-preprocess/` | InternData-A1(Franka **仿真**,单臂 7-DOF) | lerobot **v3.0**(视频 AV1、超长打包) | **AV1→h264 转码 + ≤960s 打包**;eef 用 `gripper.pose` 欧拉角 + 标准 Franka FK 交叉核验 |

## 统一工作流(各项目同构)

```
① 解压(仅压缩包交付的数据集)  →  ② 转换  →  ③ 全量校验(必须 ALL CHECKS PASSED)
   extract.py                      convert.py      validate.py
```

每个项目的 README 包含:**源数据诊断 / 55D 映射决策 / 输出结构 / 批量执行(逐步命令) /
验证结果 / 已知问题**。批量场景(一个文件夹下多个压缩包)直接看对应 README 的
"批量执行"章节。

## 迁到其他环境时的要点

- **显式传路径**:运行用 `--src-root` / `--out`(脚本内默认值指向本机历史路径,仅为方便);
- **依赖**:`pyarrow numpy scipy`(extract 脚本仅用标准库);InternData 项目的视频转码
  另需 `ffmpeg` / `ffprobe` 在 PATH;`av` / `lerobot` 可选(校验软检查);
- **`urdf/` 目录必须与脚本一起搬**(gm100 / lingbot 的 convert 依赖它做 FK;脚本通过
  **自身所在目录**解析模块与 urdf,不依赖工作目录——整目录拷贝即可用);
- **运行建议**:`cd` 进对应项目目录执行;长任务用 `setsid nohup ... > log 2>&1 &`
  (批量命令各项目 README 已给全);
- **已知问题先读**:各项目 README 的"已知数据特征 / 兼容性"章节记录了
  (如 **lerobot 0.4.4 取帧 subtask quirk**、源数据四元数缺陷编码、ffmpeg 关键帧表达式坑),
  使用前建议通读,避免踩重现的坑。

## 维护约定

- 新增数据集处理 = 在 `data_preprocess/` 下新建 `<数据集名>-preprocess/` 项目,
  沿用"extract(如需)→ convert → validate + README"的同构结构;
- 只提交代码与文档;数据/视频/日志由外层 `.gitignore` 统一排除,请勿强推大文件。
