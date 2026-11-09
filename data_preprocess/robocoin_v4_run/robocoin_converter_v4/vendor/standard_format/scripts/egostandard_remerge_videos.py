#!/usr/bin/env python
"""Merge per-episode mp4s of a canonical_55d dataset into ~500MB rotated files.

The EgoStandard chain (v30 passthrough) hardlinks one mp4 per episode into the
output. At 50TB scale that is far too many small files; this tool merges them
with ffmpeg concat -c copy (all videos are uniformly encoded h264 960x728 by
egostandard_staging.py, so stream copy is exact — no re-encode).

Per dataset:
  1. read meta/episodes parquet (episode order + lengths + current video loc)
  2. per camera: group episodes into <=500MB batches, concat -> chunk-XXX/
     file-YYY.mp4 (official 1000-files-per-chunk semantics), from/to_timestamp
     = cumulative episode durations (n_frames/fps, container time axis)
  3. verify merged nb_frames == sum(episode lengths) (ffprobe)
  4. rewrite episodes parquet video loc columns (all other columns preserved)
  5. atomic swap videos/ -> merged tree, delete per-episode files
  6. touch .remerged marker (idempotent; existing marker = skip)

Usage:
  egostandard_remerge_videos.py <out_root> [--workers 3]
Run under ionice -c3 nice -n 19. task-monitor generic log format on stdout.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FFMPEG = "/home/z00827204/miniconda3/envs/lerobot/bin/ffmpeg"
FFPROBE = "/home/z00827204/miniconda3/envs/lerobot/bin/ffprobe"
VIDEO_FILE_SIZE_MB = 500
CHUNK_SIZE = 1000
FPS = 30.0


def log(line: str) -> None:
    print(line, flush=True)


def _probe_frames(path: Path) -> int:
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    return int(r.stdout.strip())


def remerge_one(canon: Path) -> None:
    marker = canon / ".remerged"
    if marker.exists():
        return
    info = __import__("json").loads((canon / "meta" / "info.json").read_text())
    fps = float(info.get("fps", FPS))
    cams = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")

    ep_files = sorted((canon / "meta" / "episodes").glob("chunk-*/*.parquet"))
    tabs = [pq.read_table(str(f)) for f in ep_files]
    ep_tbl = pa.concat_tables(tabs)
    rows = ep_tbl.to_pylist()
    rows.sort(key=lambda r: r["episode_index"])

    merged_root = canon / "videos_merged"
    if merged_root.exists():
        import shutil
        shutil.rmtree(merged_root)

    for cam in cams:
        ck, fk = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
        ft, tt = f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"
        chunk_idx = file_idx = 0
        acc: list[dict] = []
        acc_size = 0.0

        def flush() -> None:
            nonlocal chunk_idx, file_idx, acc, acc_size
            if not acc:
                return
            out_path = merged_root / cam / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            list_f = out_path.parent / f"_concat_{file_idx:03d}.txt"
            list_f.write_text("".join(
                "file '{}'\n".format(str(
                    canon / 'videos' / cam / f'chunk-{r[ck]:03d}' / f'file-{r[fk]:03d}.mp4'
                ).replace("'", "'\\''"))  # ffmpeg concat single-quote escaping
                for r in acc))
            try:
                subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error",
                                "-f", "concat", "-safe", "0", "-i", str(list_f),
                                "-c", "copy", "-y", str(out_path)],
                               check=True, timeout=7200, capture_output=True, text=True)
            finally:
                list_f.unlink(missing_ok=True)
            expect = sum(r["length"] for r in acc)
            got = _probe_frames(out_path)
            if abs(got - expect) > max(2, expect * 3e-5):
                raise RuntimeError(f"{cam} file-{file_idx:03d} frames {got} != {expect}")
            t = 0.0
            for r in acc:
                d = r["length"] / fps
                r[ck], r[ft], r[tt] = chunk_idx, t, t + d
                r[fk] = file_idx
                t += d
            file_idx += 1
            if file_idx == CHUNK_SIZE:
                chunk_idx, file_idx = chunk_idx + 1, 0
            acc, acc_size = [], 0.0

        for r in rows:
            src = canon / "videos" / cam / f"chunk-{r[ck]:03d}" / f"file-{r[fk]:03d}.mp4"
            sz = src.stat().st_size
            if acc and acc_size + sz >= VIDEO_FILE_SIZE_MB * 1e6:
                flush()
            acc.append(r)
            acc_size += sz
        flush()

    # rewrite episodes parquet with updated video loc columns (all else preserved)
    new_tbl = pa.Table.from_pylist(rows, schema=ep_tbl.schema)
    (canon / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(new_tbl, canon / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                   compression="zstd")
    for f in ep_files:
        if f.parent.name != "chunk-000" or f.name != "file-000.parquet":
            f.unlink()

    # atomic-ish swap: old per-episode tree removed only after success
    import shutil
    old = canon / "videos"
    trash = canon / "videos_per_episode_old"
    if trash.exists():
        shutil.rmtree(trash)
    old.rename(trash)
    merged_root.rename(old)
    shutil.rmtree(trash)
    marker.write_text("ok\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-age-min", type=float, default=0.0,
                    help="skip datasets whose meta/info.json is younger than this "
                         "(avoid racing a concurrent conversion driver)")
    args = ap.parse_args()

    canons = sorted(d for d in Path(args.out_root).iterdir()
                    if d.is_dir() and (d / "meta" / "info.json").exists())
    cutoff = time.time() - args.min_age_min * 60
    todo = [c for c in canons if not (c / ".remerged").exists()
            and (c / "meta" / "info.json").exists()
            and (c / "meta" / "info.json").stat().st_mtime < cutoff]
    log(f"TASK_START total={len(todo)}")

    def work(item: tuple[int, Path]) -> None:
        i, canon = item
        t0 = time.perf_counter()
        name = canon.name
        try:
            remerge_one(canon)
            log(f"[{i}/{len(todo)}] {name} OK {time.perf_counter() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            log(f"[{i}/{len(todo)}] {name} FAIL {time.perf_counter() - t0:.1f}s {e}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, enumerate(todo, 1)))
    log("TASK_DONE")


if __name__ == "__main__":
    sys.exit(main())
