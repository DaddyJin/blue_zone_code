# URDF 说明（Piper 臂运动学模型）

本目录只保留一个文件：`aloha_tracer2_dabai_dark.urdf` —— 用于把 GM100 关节角
正运动学（FK）换算成末端位姿（eef）的机械臂模型。

## 目录

- [来源](#来源)
- [保留 / 删除](#保留--删除)
- [机械臂结构速记](#机械臂结构速记)
  - [使用前提（重要）](#使用前提重要)

## 来源

| 项 | 内容 |
|---|---|
| 仓库 | https://github.com/agilexrobotics/mobile_aloha_sim |
| commit | `f799ae0192c0c2bbd502ec4d9bbec8c2633acd1a`（2026-04-30，"统一collision和visual"） |
| 原路径 | `aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf` |
| 选择依据 | 仓库 README 指定执行 `display_aloha_tracer2_dabai.launch`，该 launch 加载的就是此文件；且已验证各 URDF 变体的臂链条参数完全一致 |

## 保留 / 删除

- **保留**：该 URDF（纯 XML；每条 joint 的 `origin`/`axis` 齐全，运动学自包含）
- **删除**：仓库其余全部内容（其他 URDF 变体、meshes ~390MB、launch/rviz、.git 历史）
- **理由**：做 FK 只需要 XML 里的运动学参数，不需要 mesh 几何
- **注意**：若将来需要在 RViz / 仿真里**可视化**该模型，需重新克隆仓库
  （URDF 内的 `package://aloha_new_description/...` 和 `package://tracer2_description/...`
  共 99 处 mesh 引用，当前均已无对应文件）

## 机械臂结构速记

- 整机 = Tracer2 移动底盘 + **4 条 Piper 臂**：
  - `fl` / `fr` = **slave（工作臂）**，fl = 左臂、fr = 右臂（按安装位置 y=±0.3 判断）
  - `bl` / `br` = master（遥操作臂，采集时人手操控用）
- 每条臂链条：`X_base_joint`(fixed) → `X_link1..6`（6 个 revolute）→ `X_joint7/8`（prismatic 双指夹爪）
- FK 输入：`X_joint1..6` 的关节角；输出：`X_link6` 的位姿（= eef）

### 使用前提（重要）

该 URDF 是 **mobile ALOHA 的 Piper 臂**；GM100 数据集的机器人为 `songling_selfcollect`。
**使用前须确认 GM100 实体臂确实为 Piper 臂**（关节数 6 + 双指夹爪结构高度吻合，待硬件确认）。
即使型号一致，仍需对齐：关节顺序映射、零位偏移、符号方向。
