#!/usr/bin/env python
"""Batch-extract dataset archives (*.tar.gz) into lerobot dataset dirs.

Generic extractor (same tool as the lingbot project): at scale the input
arrives as many compressed archives; each archive contains one (or more)
lerobot dataset dirs:

    <src-root>/.../close_the_laptop.tar.gz  ->  <dest>/close_the_laptop/
                                                  ├── data/chunk-000/file-*.parquet
                                                  ├── videos/<key>/chunk-000/*.mp4
                                                  └── meta/{info.json,...}

Design (batch-safe):
  - scans --src-root recursively for *.tar.gz / *.tgz (skips hidden dirs)
  - extracts into a hidden temp dir first, verifies each top-level dir looks
    like a lerobot dataset (meta/info.json + data/ + videos/), then moves the
    dirs into place -> interruptions never leave a half-extracted dataset
  - idempotent / resumable: already extracted & valid archives are [SKIP]ed;
    --force re-extracts from scratch
  - --continue-on-error keeps going after a bad archive; failed ones listed
  - never deletes or modifies the source archives; extraction is in place
    (next to each archive) by default, override with --dest

Usage (cosmos-framework env; needs only the stdlib):
  python intern_extract.py                       # default src-root (this project)
  python intern_extract.py --src-root /path/to/archives-root
  python intern_extract.py --src-root ... --dest /path/to/extracted
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
from pathlib import Path


def log(line: str) -> None:
    print(line, flush=True)


def find_archives(src_root: Path) -> list[Path]:
    """All *.tar.gz / *.tgz under src_root (recursive; hidden dirs skipped)."""
    out: list[Path] = []
    for p in sorted(src_root.rglob("*")):
        if not p.is_file() or not (p.name.endswith(".tar.gz") or p.name.endswith(".tgz")):
            continue
        if any(part.startswith(".") for part in p.relative_to(src_root).parts[:-1]):
            continue
        out.append(p)
    return out


def top_dirs_of(tf: tarfile.TarFile) -> list[str]:
    """Top-level directory names inside the archive (normalized, sorted)."""
    tops: set[str] = set()
    for m in tf.getmembers():
        name = m.name
        while name.startswith("./"):
            name = name[2:]
        parts = name.split("/")
        if parts and parts[0]:
            tops.add(parts[0])
    return sorted(tops)


def dataset_ok(d: Path) -> bool:
    """A lerobot dataset dir: has meta/info.json + data/ + videos/."""
    return ((d / "meta" / "info.json").is_file()
            and (d / "data").is_dir()
            and (d / "videos").is_dir())


def extract_archive(archive: Path, dest_root: Path, force: bool) -> str:
    """Extract one archive. Returns 'ok' | 'skip'. Raises on failure."""
    with tarfile.open(archive, "r:gz") as tf:
        tops = top_dirs_of(tf)
        if not tops:
            raise RuntimeError("archive has no top-level entries")

        # idempotent skip: every contained dataset dir already extracted & valid
        if not force and all(dataset_ok(dest_root / t) for t in tops):
            return "skip"

        dest_root.mkdir(parents=True, exist_ok=True)
        # unique temp name: same-named archives from different folders may share
        # one --dest in batch runs; hash the full path to avoid collisions
        tag = hashlib.md5(str(archive.resolve()).encode()).hexdigest()[:8]
        tmp = dest_root / f".extracting_{archive.name}.{tag}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            try:
                tf.extractall(tmp, filter="data")      # py>=3.12 safe extraction
            except TypeError:                          # older python: no filter kw
                tf.extractall(tmp)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        # verify every top dir looks like a lerobot dataset before publishing
        for t in tops:
            if not dataset_ok(tmp / t):
                shutil.rmtree(tmp, ignore_errors=True)
                raise RuntimeError(
                    f"extracted '{t}' is not a lerobot dataset "
                    f"(missing meta/info.json or data/videos); archive layout unexpected")

        # move into place; keep pre-existing valid dirs, replace invalid ones
        for t in tops:
            final = dest_root / t
            if final.exists():
                if not force and dataset_ok(final):
                    continue                               # keep existing valid copy
                shutil.rmtree(final)
            shutil.move(str(tmp / t), str(final))
        shutil.rmtree(tmp, ignore_errors=True)
        return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="/data5T/liujin/dataset/InternData-A1",
                    help="root scanned recursively for *.tar.gz / *.tgz archives")
    ap.add_argument("--dest", default=None,
                    help="extraction root; default: each archive is extracted in place "
                         "(next to the archive)")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the dataset dirs already exist")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going after a bad archive; failed archives are listed at the end")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    if not src_root.is_dir():
        log(f"src-root not found: {src_root}")
        sys.exit(1)
    archives = find_archives(src_root)
    if not archives:
        log(f"no *.tar.gz / *.tgz archives found under {src_root}")
        sys.exit(1)

    log(f"EXTRACT_START archives={len(archives)} src={src_root}")
    n_ok = n_skip = 0
    failed: list[str] = []
    for a in archives:
        dest_root = Path(args.dest) if args.dest else a.parent
        t0 = time.time()
        try:
            r = extract_archive(a, dest_root, args.force)
        except Exception as e:  # noqa: BLE001
            log(f"[FAIL] {a}: {type(e).__name__}: {e}")
            failed.append(str(a))
            if not args.continue_on_error:
                raise
            continue
        if r == "skip":
            n_skip += 1
            log(f"[SKIP] {a}: already extracted (use --force to redo)")
        else:
            n_ok += 1
            size_mb = a.stat().st_size / 1e6
            log(f"[OK]   {a}: {size_mb:.0f} MB archive -> {dest_root} ({time.time() - t0:.1f}s)")

    log(f"EXTRACT_DONE ok={n_ok} skip={n_skip} failed={len(failed)}")
    if failed:
        log("FAILED_ARCHIVES:")
        for f in failed:
            log(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
