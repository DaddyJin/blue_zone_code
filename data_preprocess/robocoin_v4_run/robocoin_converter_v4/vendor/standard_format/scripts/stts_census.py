#!/usr/bin/env python
"""快速普查视频文件 pts 健康度: 直接解析 mp4 moov 里的 stts 盒(不读 mdat,
每文件只读几 KB), 统计时间戳间隔异常的视频样本数。

健康: 所有样本 delta 相同(如 1/15360 下 512 tick = 1/30s)。
受损: 存在异常 delta(p2 concat 产生的 400 tick / 1 tick / 0 tick 等)。

输出: JSON {file: {"samples":N, "bad":M, "dom":D}} 仅记录 bad>0 的文件 + error。
用法: stts_census.py <task_list.json|all2002> [--workers 16]
"""
import json
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/disk2/egostandard_55d")
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def find_stts(f, pos, end):
    """在 [pos,end) 区间递归找 stts 盒, 返回其数据起点偏移。"""
    f.seek(pos)
    while f.tell() + 8 <= end:
        start = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            return None
        size, typ = struct.unpack(">I4s", hdr)
        hdr_len = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr_len = 16
        elif size == 0:
            size = end - start
        if size < hdr_len or start + size > end:
            return None
        if typ == b"stts":
            return start + hdr_len
        if typ in CONTAINERS:
            r = find_stts(f, start + hdr_len, start + size)
            if r is not None:
                return r
        f.seek(start + size)
    return None


def stts_stats(path: Path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        fsize = f.tell()
        off = find_stts(f, 0, fsize)
        if off is None:
            return None
        f.seek(off + 4)  # version/flags
        n = struct.unpack(">I", f.read(4))[0]
        if n > 10_000_000:
            return None
        data = f.read(n * 8)
    if len(data) < n * 8:
        return None
    deltas = Counter()
    total = 0
    for i in range(n):
        count, delta = struct.unpack(">II", data[i*8:i*8+8])
        deltas[delta] += count
        total += count
    dom, dom_n = deltas.most_common(1)[0]
    return total, total - dom_n, dom


def scan_file(p: Path):
    try:
        r = stts_stats(p)
        if r is None:
            return str(p), {"error": "no stts"}
        total, bad, dom = r
        if bad > 0:
            return str(p), {"samples": total, "bad": bad, "dom": dom}
    except Exception as e:  # noqa: BLE001
        return str(p), {"error": str(e)[:100]}
    return None


def main():
    workers = 16
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    list_json = sys.argv[1]
    if list_json == "all2002":
        tasks = json.load(open("/home/z00827204/hf_tools/body_complete_tasks.json"))
    else:
        tasks = json.load(open(list_json))
    files = []
    for t in tasks:
        files += sorted((ROOT / t / "videos").glob("*/chunk-*/*.mp4"))
    print(f"TASK_START total={len(files)}", flush=True)
    out = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(scan_file, p) for p in files]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r:
                out[r[0]] = r[1]
            if i % 2000 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] bad_files={len(out)} {time.time()-t0:.0f}s",
                      flush=True)
    json.dump(out, open("/home/z00827204/hf_tools/stts_census_body2002.json", "w"), indent=0)
    print(f"TASK_DONE bad_files={len(out)}", flush=True)


if __name__ == "__main__":
    main()
