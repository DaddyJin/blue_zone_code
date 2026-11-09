#!/usr/bin/env python
"""A 路径修复: 就地重写 mp4 时间戳表, 把原生帧率时间轴改为 30fps 容器时间轴。

原理: stts/ctts/stss 等表里的数值都是 "帧序号 × 网格步长"。
原生 25fps: timebase 12800, 步长 512(=0.04s)。
目标 30fps: timebase 15360, 步长 512(=1/30s)。
=> 网格整数关系不变, 只需把 mdhd timescale 12800->15360,
   并按比例修正 mdhd/tkhd/mvhd 的 duration 字段。
mixed 文件(多网格): stts 中 25fps 段 delta=512@12800, 30fps 段 delta=427/426@12800
   等; 需要把所有段的 delta 统一改为 512@15360, ctts 同步按段网格换算。
   (v1 先只支持 uniform; mixed 走单独分支)

用法: mp4_timescale_fix.py <file.mp4> [--write]
  不加 --write 只诊断不修改。修改先生成 .fixed 再原子替换, 并保留 .bak。
"""
import shutil
import struct
import sys
from pathlib import Path

TARGET_TIMESCALE = 15360  # 512 ticks = 1/30 s
TARGET_DELTA = 512


def _read_boxes(data, start, end):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", data, pos)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        yield typ, pos, size, hdr
        pos += size


def _find(data, start, end, want):
    for typ, pos, size, hdr in _read_boxes(data, start, end):
        if typ == want:
            return pos, size, hdr
    return None


def analyze(path):
    data = open(path, "rb").read()
    moov = _find(data, 0, len(data), b"moov")
    assert moov, "no moov"
    trak = _find(data, moov[0] + moov[2], moov[0] + moov[1], b"trak")
    mdia = _find(data, trak[0] + trak[2], trak[0] + trak[1], b"mdia")
    mdhd = _find(data, mdia[0] + mdia[2], mdia[0] + mdia[1], b"mdhd")
    pos, size, hdr = mdhd
    ver = data[pos + hdr]
    off = pos + hdr + 4
    if ver == 1:
        off += 16
    else:
        off += 8
    timescale, duration = struct.unpack_from(">II", data, off)
    minf = _find(data, mdia[0] + mdia[2], mdia[0] + mdia[1], b"minf")
    stbl = _find(data, minf[0] + minf[2], minf[0] + minf[1], b"stbl")
    stts = _find(data, stbl[0] + stbl[2], stbl[0] + stbl[1], b"stts")
    pos, size, hdr = stts
    n = struct.unpack_from(">I", data, pos + hdr + 4)[0]
    entries = []
    p = pos + hdr + 8
    for _ in range(n):
        c, d = struct.unpack_from(">II", data, p)
        entries.append((c, d))
        p += 8
    total_samples = sum(c for c, _ in entries)
    return {"timescale": timescale, "duration": duration,
            "stts": entries, "samples": total_samples,
            "native_fps": round(timescale / entries[0][1], 3)}


def fix_uniform(path, write=False):
    """uniform 文件: stts 全部同一 delta。把所有帧的呈现间隔统一为 1/30s:
    保持 tick 数值不变, 只把 timescale 改为 delta*30 (delta=512 -> 15360)。
    内容会被整体变速到 30fps (25fps 源加速, 36fps 源减速) — 这是既定策略。"""
    data = bytearray(open(path, "rb").read())
    a = analyze(path)
    deltas = {d for _, d in a["stts"]}
    assert len(deltas) == 1, f"不是 uniform: {deltas}"
    delta0 = deltas.pop()
    ts0 = a["timescale"]
    target_ts = delta0 * 30
    n = a["samples"]
    moov = _find(data, 0, len(data), b"moov")
    trak = _find(data, moov[0] + moov[2], moov[0] + moov[1], b"trak")
    mdia = _find(data, trak[0] + trak[2], trak[0] + trak[1], b"mdia")
    mdhd = _find(data, mdia[0] + mdia[2], mdia[0] + mdia[1], b"mdhd")
    pos, size, hdr = mdhd
    ver = data[pos + hdr]
    off = pos + hdr + 4 + (16 if ver == 1 else 8)
    old_dur_ticks = struct.unpack_from(">I", data, off + 4)[0]
    new_dur_ticks = n * delta0
    # tkhd duration(movie timescale, 通常 1000)
    tkhd = _find(data, trak[0] + trak[2], trak[0] + trak[1], b"tkhd")
    tpos, tsize, thdr = tkhd
    tver = data[tpos + thdr]
    doff = tpos + thdr + 4 + (20 if tver == 1 else 12)
    # mvhd
    mvhd = _find(data, moov[0] + moov[2], moov[0] + moov[1], b"mvhd")
    mpos, msize, mhdr = mvhd
    mver = data[mpos + mhdr]
    mts_off = mpos + mhdr + 4 + (16 if mver == 1 else 8)
    movie_ts = struct.unpack_from(">I", data, mts_off)[0]
    new_dur_movie = round(n / 30.0 * movie_ts)
    # elst: segment_duration 也在 movie timescale 下, 不同步会让 ffprobe
    # 按旧时长报告 (media_time 是 media ticks, 数值不变不用改)
    elst_edits = []  # (offset, size_fmt)
    edts = _find(data, trak[0] + trak[2], trak[0] + trak[1], b"edts")
    if edts:
        elst = _find(data, edts[0] + edts[2], edts[0] + edts[1], b"elst")
        if elst:
            epos, esize, ehdr = elst
            ever = data[epos + ehdr]
            ecnt = struct.unpack_from(">I", data, epos + ehdr + 4)[0]
            eoff = epos + ehdr + 8
            for _ in range(ecnt):
                mt_off = eoff + (8 if ever == 1 else 4)
                mt = struct.unpack_from(">q" if ever == 1 else ">i",
                                        data, mt_off)[0]
                if mt >= 0:
                    elst_edits.append((eoff, ">Q" if ever == 1 else ">I"))
                eoff += 20 if ever == 1 else 12
    # 判断是否有任何字段需要改动 (幂等: 全部已正确则返回 None)
    cur_mdhd_ts = struct.unpack_from(">I", data, off)[0]
    cur_mdhd_dur = old_dur_ticks
    cur_tkhd_dur = struct.unpack_from(">I", data, doff)[0]
    cur_mvhd_dur = struct.unpack_from(">I", data, mts_off + 4)[0]
    cur_elst = [struct.unpack_from(fmt, data, o)[0] for o, fmt in elst_edits]
    needs = (cur_mdhd_ts != target_ts or cur_mdhd_dur != new_dur_ticks
             or cur_tkhd_dur != new_dur_movie or cur_mvhd_dur != new_dur_movie
             or any(v != new_dur_movie for v in cur_elst))
    info = {"old_timescale": ts0, "new_timescale": target_ts,
            "old_duration_ticks": old_dur_ticks, "new_duration_ticks": new_dur_ticks,
            "native_fps": ts0 / delta0, "samples": n, "elst_patched": len(elst_edits)}
    if not needs:
        return None  # 已是 30fps 网格且所有时长字段一致, 无需修
    if not write:
        return info
    struct.pack_into(">I", data, off, target_ts)
    struct.pack_into(">I", data, off + 4, new_dur_ticks)
    struct.pack_into(">I", data, doff, new_dur_movie)
    struct.pack_into(">I", data, mts_off + 4, new_dur_movie)
    for o, fmt in elst_edits:
        struct.pack_into(fmt, data, o, new_dur_movie)
    out = path.with_suffix(".mp4.fixed")
    open(out, "wb").write(bytes(data))
    return info


if __name__ == "__main__":
    p = Path(sys.argv[1])
    if "--write" in sys.argv:
        info = fix_uniform(p, write=True)
        if info is None:
            print("SKIP already 30fps grid")
        else:
            fixed = p.with_suffix(".mp4.fixed")
            bak = p.with_suffix(".mp4.bak")
            p.rename(bak)
            fixed.rename(p)
            bak.unlink()
            print("FIXED", info)
    else:
        print(analyze(p))
