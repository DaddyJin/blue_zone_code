#!/usr/bin/env python
"""durcap 重轮转 v2: 按文件切分 + 安全文件硬链接, IO 只花在长文件上。

背景见 v1 头部注释(lerobot 0.5.0 torchcodec float32 容差, 文件必须 <1024s)。
v2 与 v1 的差异: v1 全局装箱重写任务全部视频(~2.7TB); v2 只处理 >960s 的
长文件(~1.2TB), <=960s 的文件硬链接进新树, from/to 不变。

每相机:
  1. 按 file_index 顺序遍历旧文件, ffprobe 时长
  2. <=CAP: 硬链接为新序号文件, 其中 episode 的 from/to 保持不变
     >CAP: 该文件的 episode 按 <=CAP 装箱(最大 711s, 切点都在 episode 边界),
            PyAV 单输入多输出重打包, pts/dts 整数 tick 重基
  3. 新文件统一顺序编号(chunk-000); 重写 episodes parquet 视频定位列
  4. 新文件 ffprobe 帧数校验; 旧目录确认后删除; 落 .durcap_done 标记(幂等)

用法: egostandard_durcap_rotate2.py <task_dir>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq

FFPROBE = "/home/z00827204/miniconda3/envs/lerobot/bin/ffprobe"
CAP_S = 960.0
FPS = 30.0
MARKER = ".durcap_done"


def _probe(path: Path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames:format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    nf, dur = int(lines[0].split(",")[0]), float(lines[-1].split(",")[-1])
    return dur, nf


def rotate_cam(task: Path, cam: str, rows, tmp_root: Path):
    """处理一个相机, 返回 {episode_index: (file_index, from_s, to_s)}。"""
    ck = f"videos/{cam}/chunk_index"
    fk = f"videos/{cam}/file_index"
    ft = f"videos/{cam}/from_timestamp"
    tt = f"videos/{cam}/to_timestamp"
    # 旧文件 -> 其 episode(按 from 排序)
    by_old: dict[int, list] = {}
    for r in rows:
        by_old.setdefault(r[fk], []).append(r)
    out_dir = tmp_root / cam / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    loc = {}
    new_idx = 0
    for old_fk in sorted(by_old):
        eps = sorted(by_old[old_fk], key=lambda r: r[ft])
        old_file = task / "videos" / cam / f"chunk-{eps[0][ck]:03d}" / f"file-{old_fk:03d}.mp4"
        dur, _ = _probe(old_file)
        if dur <= CAP_S:
            # 硬链接, from/to 不变
            os_link = out_dir / f"file-{new_idx:03d}.mp4"
            os_link.hardlink_to(old_file)
            for r in eps:
                loc[r["episode_index"]] = (new_idx, r[ft], r[tt])
            new_idx += 1
            continue
        # 长文件: 按 <=CAP 装箱
        packs, cur, acc = [], [], 0.0
        for r in eps:
            d = r[tt] - r[ft]
            if cur and acc + d > CAP_S:
                packs.append(cur)
                cur, acc = [], 0.0
            cur.append(r)
            acc += d
        if cur:
            packs.append(cur)
        # PyAV 单输入多输出
        c_in = av.open(str(old_file))
        v_in = c_in.streams.video[0]
        tb = v_in.time_base
        ticks_per_frame = round((1 / FPS) / tb)
        # pack -> (new_idx, 起始帧 offset)
        plan = []
        pack_nfs = []
        for pack in packs:
            pack_nfs.append(new_idx)
            off = 0
            for r in pack:
                plan.append((r, new_idx, off))
                off += int(round((r[tt] - r[ft]) * FPS))
            new_idx += 1
        outs, out_streams, written = {}, {}, {}
        ranges = []  # (t0, t1, new_idx, base_frames, row)
        for r, nf, off in plan:
            t0 = round(r[ft] / tb)
            t1 = round(r[tt] / tb)
            ranges.append((t0, t1, nf, off, r))
        if ranges:
            # 尾部开放: 末集 t1 之后仍可能有少量尾包(容器时长略超元数据),
            # 丢弃会造成帧数差(2026-10-08 Dressing_Table 差2帧失败案例)
            _t0l, _t1l, _nfl, _offl, _rl = ranges[-1]
            ranges[-1] = (_t0l, 1 << 62, _nfl, _offl, _rl)
        for pkt in c_in.demux(v_in):
            if pkt.pts is None:
                continue
            for t0, t1, nf, off, r in ranges:
                if t0 <= pkt.pts < t1:
                    if nf not in outs:
                        p = out_dir / f"file-{nf:03d}.mp4"
                        c = av.open(str(p), mode="w")
                        outs[nf] = c
                        out_streams[nf] = c.add_stream_from_template(v_in)
                        written[nf] = 0
                    base = off * ticks_per_frame
                    pkt.pts = pkt.pts - t0 + base
                    if pkt.dts is not None:
                        pkt.dts = pkt.dts - t0 + base
                    pkt.stream = out_streams[nf]
                    outs[nf].mux(pkt)
                    written[nf] += 1
                    break
        c_in.close()
        for nf, c in outs.items():
            c.close()
        # 校验 + 记录定位
        for pack, nf0 in zip(packs, pack_nfs):
            expect = sum(int(round((r[tt] - r[ft]) * FPS)) for r in pack)
            p = out_dir / f"file-{nf0:03d}.mp4"
            got = _probe(p)[1]
            diff = got - expect
            if abs(diff) > 10:
                raise RuntimeError(f"{cam} file-{nf0:03d} frames {got} != {expect}")
            # 个位数帧差(尾包/重复 pts)归入 pack 内最后一集, 保证索引真实
            if diff:
                print(f"  [durcap] {cam} file-{nf0:03d} 帧差 {diff:+d} 归入末集",
                      flush=True)
            off = 0
            for j, r in enumerate(pack):
                nf_r = int(round((r[tt] - r[ft]) * FPS))
                if j == len(pack) - 1:
                    nf_r += diff
                loc[r["episode_index"]] = (nf0, off / FPS, (off + nf_r) / FPS)
                off += nf_r
    return loc


def durcap_one(task: Path) -> str:
    if (task / MARKER).exists():
        return "skip"
    info = json.loads((task / "meta" / "info.json").read_text())
    cams = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    # 是否需要
    needs = False
    for cam in cams:
        for f in (task / "videos" / cam).glob("chunk-*/*.mp4"):
            if _probe(f)[0] > CAP_S:
                needs = True
                break
        if needs:
            break
    if not needs:
        (task / MARKER).write_text("not-needed\n")
        return "not-needed"

    ep_files = sorted((task / "meta" / "episodes").glob("chunk-*/*.parquet"))
    tbl = pa.concat_tables([pq.read_table(str(f)) for f in ep_files])
    rows = tbl.to_pylist()
    rows.sort(key=lambda r: r["episode_index"])

    tmp_root = task / "videos_durcap"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    locs = {cam: rotate_cam(task, cam, rows, tmp_root) for cam in cams}

    for r in rows:
        for cam in cams:
            nf, fs, ts = locs[cam][r["episode_index"]]
            r[f"videos/{cam}/chunk_index"] = 0
            r[f"videos/{cam}/file_index"] = nf
            r[f"videos/{cam}/from_timestamp"] = fs
            r[f"videos/{cam}/to_timestamp"] = ts
    new_tbl = pa.Table.from_pylist(rows, schema=tbl.schema)
    out_ep = task / "meta" / "episodes" / "chunk-000"
    out_ep.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_tbl, out_ep / "file-000.parquet.tmp", compression="zstd")
    (out_ep / "file-000.parquet.tmp").rename(out_ep / "file-000.parquet")
    for f in ep_files:
        if f.parent.name != "chunk-000" or f.name != "file-000.parquet":
            f.unlink()

    old = task / "videos"
    trash = task / "videos_pre_durcap"
    if trash.exists():
        shutil.rmtree(trash)
    old.rename(trash)
    tmp_root.rename(old)
    shutil.rmtree(trash)
    (task / MARKER).write_text("ok\n")
    return "ok"


def main():
    task = Path(sys.argv[1])
    print(f"{task.name}: {durcap_one(task)}", flush=True)


if __name__ == "__main__":
    main()
