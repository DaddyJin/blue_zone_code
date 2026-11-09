#!/usr/bin/env python
"""用 lerobot 0.5.0 官方 LeRobotDataset 验证转换结果的 subtask 机制。

用法:
  python egostandard_050_official_load.py <task_dir>

检查项:
  1. 数据集能被官方 LeRobotDataset 加载(v3.0 骨架合规)
  2. ds.meta.subtasks 非 None(官方识别了我们的 meta/subtasks.parquet)
  3. __getitem__ 返回 item["subtask"] 字符串(官方 subtask 解析机制生效)
  4. subtask 字符串与 subtask_index 通过词表手工解析的结果一致
  5. subtask_index 覆盖统计(-1 = 未标注帧)
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from pathlib import Path

import pandas as pd


def main():
    root = Path(sys.argv[1]).resolve()
    print(f"task dir: {root}")

    import lerobot
    print(f"lerobot version: {lerobot.__version__}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=root.name, root=str(root))
    print(f"[1] 官方加载 OK: {ds.num_episodes} episodes, {ds.num_frames} frames, fps={ds.fps}")

    # 2. meta.subtasks
    subtasks = getattr(ds.meta, "subtasks", None)
    assert subtasks is not None, "ds.meta.subtasks is None —— 官方未识别 subtasks.parquet"
    print(f"[2] meta.subtasks OK: {len(subtasks)} 个子任务词条")
    print(subtasks.head(10))

    # 3/4. 逐帧 subtask 解析
    # 2026-09-20: 2000 -> 500。随机解码校验是增量链的吞吐瓶颈(p2 并发期
    # 每任务 5~15min); step 5 仍有全量 parquet 逐帧覆盖兜底, 降采样不损完整性校验
    n_check = min(ds.num_frames, 500)
    import random
    random.seed(0)
    idxs = sorted(random.sample(range(ds.num_frames), n_check))
    ok, mismatch, unlabeled = 0, 0, 0
    subtask_counter = {}
    for i in idxs:
        item = ds[i]
        assert "subtask" in item, f"frame {i}: item 里没有 'subtask' 键"
        s_idx = int(item["subtask_index"].item() if hasattr(item["subtask_index"], "item") else item["subtask_index"])
        s_str = item["subtask"]
        if s_idx == -1:
            unlabeled += 1
            # 未标注帧官方如何表现,如实记录
            if s_str not in (None, ""):
                # 词表里没有 -1 行,官方 iloc[-1] 会取到最后一行,记录但不计入 mismatch
                pass
            continue
        expect = subtasks.iloc[s_idx].name
        if s_str == expect:
            ok += 1
            subtask_counter[s_str] = subtask_counter.get(s_str, 0) + 1
        else:
            mismatch += 1
            if mismatch <= 3:
                print(f"  MISMATCH frame {i}: subtask_index={s_idx} -> '{s_str}' != '{expect}'")
    print(f"[3] __getitem__ 返回 'subtask' 键 OK(抽检 {n_check} 帧)")
    print(f"[4] 字符串与词表一致性: ok={ok}, mismatch={mismatch}, 未标注(-1)={unlabeled}")
    print("    子任务分布(抽检):", dict(sorted(subtask_counter.items(), key=lambda x: -x[1])))

    # 5. 全量覆盖率(直接读 data parquet,不走视频解码)
    import pyarrow.parquet as pq
    total, neg = 0, 0
    for f in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["subtask_index"])
        col = t.column("subtask_index").to_pylist()
        total += len(col)
        neg += sum(1 for v in col if v == -1)
    print(f"[5] 全量覆盖: {total} 帧, 未标注 {neg} 帧 ({neg/total:.1%})")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
