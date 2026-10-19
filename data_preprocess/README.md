# data_preprocess — 团队标准格式数据预处理工具链

本板块收录把各种源数据转换成**团队统一格式**(LeRobot v3.0 官方布局 + 55D 统一动作列)
的**转换/校验代码**。每个子目录是一个独立项目:自带 README、转换脚本、校验脚本。
输入数据均在运行环境就地提供,本仓**只收录代码**(数据/视频/日志已由外层 `.gitignore` 排除)。

团队规范材料(`standard_format/`,含 55D 槽位布局、视频规范、校验链)为内部文档,
不在本仓;各项目 README 中引用它的地方仅为出处说明,不影响运行。

## 目录

- [三个项目速览](#三个项目速览)
- [统一工作流(各项目同构)](#统一工作流各项目同构)
- [进度与大规模并行](#进度与大规模并行)
- [迁到其他环境时的要点](#迁到其他环境时的要点)
- [维护约定](#维护约定)

## 三个项目速览

| 项目目录 | 适用数据 | 源格式 | 核心能力 |
|---|---|---|---|
| `gm100-cobotmagic-lerobot-preprocess/` | GM100 cobotmagic(双臂 6-DOF ×2 +夹爪,实机) | lerobot **v2.1**(per-episode) | v2.1→v3.0+55D;Piper URDF **FK 回填 eef**;逐集硬链接;断点续跑 |
| `lingbot-GM-100-preprocess/` | lingbot GM-100(**压缩包交付**,与 GM100 同源数据) | lerobot v2.1(**tar.gz**) | **批量解压**(幂等/原子发布)+ 同上转换 |
| `InternData-A1-preprocess/` | InternData-A1(Franka **仿真**,单臂 7-DOF) | lerobot **v3.0**(视频 AV1、超长打包) | **AV1→h264 转码 + ≤960s 打包**;eef 用 `gripper.pose` 欧拉角 + 标准 Franka FK 交叉核验 |

各数据集的**逐步执行命令**见对应 README:
[GM100 §7.3](gm100-cobotmagic-lerobot-preprocess/README.md#73-执行步骤以本数据集为例) ·
[lingbot §7.2 / §7.3](lingbot-GM-100-preprocess/README.md#72-单数据集执行以-bm-001-为例) ·
[InternData §9](InternData-A1-preprocess/README.md#9-执行流程与批量指南)。

## 统一工作流(各项目同构)

```
① 解压(仅压缩包交付的数据集)  →  ② 转换  →  ③ 全量校验(必须 ALL CHECKS PASSED)
   extract.py                      convert.py      validate.py
```

每个项目的 README 包含:**源数据诊断 / 55D 映射决策 / 输出结构 / 批量执行(逐步命令) /
验证结果 / 已知问题**。批量场景(一个文件夹下多个压缩包)直接看对应 README 的
"批量执行"章节。

## 进度与大规模并行

三套工具链的 extract / convert / validate 都内置**进度 + ETA**(按帧数或压缩包加权,
默认每 10 秒一行,适合 `nohup` 日志),并支持并行:

- `--workers N` —— 并行处理 N 个数据集/压缩包(默认 8;注意 Intern 的 `--workers` 与 `--video-workers` 相乘);
- `--video-workers N` —— Intern 转码时并行 N 个 ffmpeg 源文件任务(默认 3);
- `--no-progress` / `--progress-interval SEC` —— 关闭或调整进度行。

并行只改变调度顺序:每个数据集输出目录独立,单数据集内部仍严格串行,输出格式与
串行版完全一致。进度模块 `_progress.py` 随各项目自带,无额外依赖。

**实测加速(本机 192 核 / demo 数据,输出逐字节一致)**:

| 场景 | 串行 | 并行 | 加速 |
|---|---|---|---|
| GM100 convert,8 个 task_00001 规模 | 57 s | 8 s(`--workers 8`) | **7.1x** |
| GM100 validate,8 个 task_00001 规模 | 153 s | 27 s(`--workers 8`) | **5.7x** |
| InternData 单数据集(578 eps / 14 万帧,视频转码为主) | ~13 min | 4 min 05 s(`--video-workers 3`) | **~3.0x** |

> GM100 的 8-task 基准用同规模 task 的硬链接副本构造,用于测吞吐;真实批次受磁盘
> I/O、单 task 帧数差异影响,加速比会上下浮动。InternData 的视频并行度受“相机数 ×
> 源视频文件数”限制(本 demo 只有 3 个源文件,`--video-workers >3` 不会更快)。

## 迁到其他环境时的要点

- **显式传路径**:运行用 `--src-root` / `--out`。`--out` 默认写到**脚本所在目录**的
  `processed_dataset/`(整目录迁移后仍可用);`--src-root` 默认指向本机数据集路径,
  迁移到其他环境时请显式传参;
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
