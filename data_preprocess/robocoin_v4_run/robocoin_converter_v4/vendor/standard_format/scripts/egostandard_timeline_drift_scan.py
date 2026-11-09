#!/usr/bin/env python
"""全量扫描 /mnt/disk2/egostandard_55d: 找出视频时间轴与数据索引不一致的任务。

判据: 每个相机的合并视频文件, sum(duration) 应与 total_frames/30 一致
(容差 0.5s)。不一致 = 该任务含原生非30fps打包的 round-1 视频段。
输出: /home/z00827204/hf_tools/timeline_drift_report.json
"""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

FFPROBE = "/home/z00827204/miniconda3/envs/lerobot/bin/ffprobe"
ROOT = "/mnt/disk2/egostandard_55d"
OUT = "/home/z00827204/hf_tools/timeline_drift_report.json"
FPS = 30.0
TOL = 0.5


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-select_streams",
                        "v", "-of", "json", path], capture_output=True, text=True)
    try:
        st = json.loads(r.stdout)["streams"][0]
        return float(st.get("duration", 0)), int(st.get("nb_frames", 0)), st.get("r_frame_rate", "?")
    except Exception:
        return None, None, "probe_fail"


def check_task(d):
    tdir = os.path.join(ROOT, d)
    try:
        info = json.load(open(os.path.join(tdir, "meta", "info.json")))
    except Exception as e:
        return {"task": d, "status": "error", "why": f"info.json: {e}"}
    total_frames = info.get("total_frames", 0)
    expect_s = total_frames / info.get("fps", FPS)
    cams = {}
    worst = 0.0
    for v in sorted(glob.glob(os.path.join(tdir, "videos", "*", "chunk-*", "file-*.mp4"))):
        cam = v.split("/videos/")[1].split("/")[0]
        dur, nf, rfr = probe(v)
        if dur is None:
            return {"task": d, "status": "error", "why": f"probe fail {v}"}
        c = cams.setdefault(cam, [0.0, 0])
        c[0] += dur
        c[1] += nf
    details = {}
    for cam, (dur, nf) in cams.items():
        dev = dur - nf / FPS
        details[cam] = {"video_s": round(dur, 2), "frames_s": round(nf / FPS, 2),
                        "dev_s": round(dev, 2)}
        worst = max(worst, abs(dev))
    status = "drift" if worst > TOL else "ok"
    return {"task": d, "status": status, "worst_dev_s": round(worst, 2),
            "expect_s": round(expect_s, 2), "cams": details}


def main():
    tasks = sorted(d for d in os.listdir(ROOT)
                   if os.path.isdir(os.path.join(ROOT, d)) and d.startswith("EgoStandard"))
    print(f"TASK_START total={len(tasks)}", flush=True)
    results = []
    t0 = __import__("time").time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(check_task, d): d for d in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            results.append(fu.result())
            if i % 100 == 0:
                el = __import__("time").time() - t0
                print(f"[{i}/{len(tasks)}] scanned OK {el:.0f}s", flush=True)
    drift = [r for r in results if r["status"] == "drift"]
    err = [r for r in results if r["status"] == "error"]
    json.dump({"total": len(tasks), "drift": len(drift), "error": len(err),
               "results": sorted(results, key=lambda r: -r.get("worst_dev_s", 0))},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"TASK_DONE drift={len(drift)} error={len(err)}", flush=True)
    for r in drift[:20]:
        print("DRIFT", r["task"], r["worst_dev_s"], flush=True)


if __name__ == "__main__":
    main()
