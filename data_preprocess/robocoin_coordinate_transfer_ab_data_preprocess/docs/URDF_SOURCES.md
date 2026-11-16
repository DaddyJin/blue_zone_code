# URDF 来源与授权

生成日期：2026-11-13

## 1. 来源总表

| 来源 | 下载内容 | 本地位置 | 版本/校验 | 授权 |
|---|---|---|---|---|
| [discoverse-dev/robot_descriptions](https://github.com/discoverse-dev/robot_descriptions) | Airbot MMK2、AI2 Bot2、AgiBot G1、Realman AIDA-L、Unitree G1（社区版）、AgileX Aloha/PiPER、Galaxea R1 Lite | `sources/robot_descriptions/` | commit `08a64b1dcbe8dcce78825ebe0fe1f562aa7dbff1` | Apache-2.0（`sources/robot_descriptions/LICENSE`），各包内另有 `LICENSE` |
| [DexForceAI/embodichain_data](https://hf-mirror.com/datasets/DexForceAI/embodichain_data) → `robot_assets/CobotMagicArmV4.zip` | Cobot Magic 单臂 URDF + 视觉/碰撞 mesh | `sources/embodichain/CobotMagicArmV4/` | SHA256 `05b0a541974e08afbf359d064369c49f1796c0559eca099272018e2c8c1c74d3` | EmbodiChain 代码 Apache-2.0；zip 内未附独立 LICENSE，商用前请向 AgileX 确认 |
| [LejuRobotics/kuavo-ros-opensource](https://gitee.com/leju-robot/kuavo-ros-opensource) | Kuavo 4.x 整车 URDF + mesh（8 个版本） | `sources/kuavo/` | commit `3374a712f3c889309f8061e0beacbb99ef01f904` | 见 `sources/kuavo/LICENSE` |
| [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) | `assets/g1/g1_body29_hand14.urdf` + 49 个 mesh（RoboCOIN 代码引用的 G1 版本） | `sources/xr_teleoperate/g1_body29_hand14/` | commit `817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b` | Apache-2.0（`sources/xr_teleoperate/g1_body29_hand14/LICENSE`） |

## 2. 下载方式

- GitHub 原始文件：通过 `gh-proxy.com` / `ghproxy.net` / `cdn.jsdelivr.net` 代理访问。
- HuggingFace 资产：通过 `hf-mirror.com` 下载。
- 代理只影响传输路径，文件内容未改动。
- Leju Kuavo 通过 Gitee 镜像 `gitee.com/leju-robot/kuavo-ros-opensource` 下载，commit 与 GitHub 上游对应。

## 3. 目录与符号链接

> 本包为了通过 git 传输，只复制了 FK 所需的 `.urdf` 文件；`sources/`、mesh、
> 原始 LICENSE 文件没有随包分发。完整来源、commit 和授权仍以下表为准；
> 如需重新分发 URDF，请同时保留本文件的来源/授权信息。

- `sources/` 保留 vendor 原始目录结构，**不修改**。
- `models/` 是精选视图，里面的条目是指向 `sources/` 的**符号链接**，不是副本。
  - `models/<family>/<package>` → `../../sources/.../<package>`
  - 这样 URDF 里的相对 mesh 路径（如 `../meshes/xxx.stl`）仍然可以解析。
- 归档/拷贝时如需实体文件，用 `tar -h` 或 `cp -L` 解引用符号链接。

## 4. 修改过的文件

只有 AI2 的两个文件做了 mesh 路径重写（不改运动学）：

- `models/AI2_AlphaBot_2/ai2_bot2_description/urdf/ai2_bot2_fixed.urdf`
- `models/AI2_AlphaBot_2/ai2_bot2_description/urdf/ai2_bot2_inspire_fixed.urdf`

原始 `ai2_bot2.urdf` / `ai2_bot2_inspire.urdf` 保留未动。重写规则见根 `README.md` 第 6.2 节。

## 5. 未获取到的模型

- **Galbot G1**：未找到公开可下载的 URDF/MJCF/USD；官方开发者平台 `developer.galbot.com`。
- **Tianqing A2（天擎 A2）**：厂商未公开发布；杭州软通天擎机器人科技有限公司。
- `sources/robot_descriptions/humanoid/Galbot/galbot_one_description` 是 **Galbot One**，不是 G1，不能替代。
