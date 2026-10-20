#!/usr/bin/env python
"""Batch-extract dataset archives (*.tar.gz) into lerobot dataset dirs.

Generic extractor (same tool as the lingbot project): at scale the input
arrives as many compressed archives; each archive contains one (or more)
lerobot dataset dirs:

    <src-root>/.../close_the_laptop.tar.gz  ->  <dest>/.../close_the_laptop/
                                                  ├── data/chunk-000/file-*.parquet
                                                  ├── videos/<key>/chunk-000/*.mp4
                                                  └── meta/{info.json,...}
    (with --dest the archive's path relative to --src-root is mirrored;
     without --dest the archive extracts in place, next to itself)

Design (batch-safe):
  - scans --src-root recursively for *.tar.gz / *.tgz (skips hidden dirs)
  - extracts into a hidden temp dir first, verifies each top-level dir looks
    like a lerobot dataset (meta/info.json + data/ + videos/), then moves the
    dirs into place -> interruptions never leave a half-extracted dataset
  - idempotent / resumable: already extracted & valid archives are [SKIP]ed;
    --force re-extracts from scratch
  - --continue-on-error keeps going after a bad archive; failed ones listed
  - never deletes or modifies the source archives; extraction is in place
    (next to each archive) by default. With --dest, each archive's path
    RELATIVE to --src-root is mirrored under it, e.g.
      <src-root>/AgileX/BM-001.tar.gz  --dest D  ->  D/AgileX/BM-001/

Usage (cosmos-framework env; needs only the stdlib):
  python intern_extract.py                       # default src-root (this project)
  python intern_extract.py --src-root /path/to/archives-root
  python intern_extract.py --src-root ... --dest /path/to/extracted
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import shutil
import sys
import tarfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _progress import Progress, log_line  # noqa: E402


def log(line: str) -> None:
    log_line(line)


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


def dest_for(archive: Path, src_root: Path, dest: Path | None) -> Path:
    """Extraction root for one archive.

    Without --dest: extract in place (next to the archive).
    With --dest: MIRROR the archive's path relative to --src-root under it,
    preserving the archive tree's directory hierarchy, e.g.
        A/AgileX/BM-001.tar.gz  --dest B  ->  B/AgileX/BM-001/
    """
    if dest is None:
        return archive.parent
    try:
        rel = archive.parent.relative_to(src_root)
    except ValueError:                      # archive not under src-root (defensive)
        rel = Path(".")
    return dest / rel


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


def _publish_lock(dest_root: Path):
    """Cross-process lock so parallel archives publishing into one --dest do not
    race on same-named top-level dataset dirs. Only the final move is locked."""
    dest_root.mkdir(parents=True, exist_ok=True)
    fh = open(dest_root / ".extract_publish.lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


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

        # move into place; keep pre-existing valid dirs, replace invalid ones.
        # Serialize only the publish step: with --workers > 1, multiple archives
        # may share one --dest and race on same-named top-level dirs.
        lock = _publish_lock(dest_root)
        try:
            for t in tops:
                final = dest_root / t
                if final.exists():
                    if not force and dataset_ok(final):
                        continue                           # keep existing valid copy
                    shutil.rmtree(final)
                shutil.move(str(tmp / t), str(final))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        shutil.rmtree(tmp, ignore_errors=True)
        return "ok"


def _extract_worker(archive: str, dest: str, force: bool) -> dict:
    """Worker-process entry point: extract one archive."""
    a = Path(archive)
    t0 = time.time()
    try:
        status = extract_archive(a, Path(dest), force)
        return {"ok": True, "status": status, "archive": archive,
                "size_mb": a.stat().st_size / 1e6, "seconds": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "archive": archive,
                "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="/data5T/liujin/dataset/InternData-A1",
                    help="root scanned recursively for *.tar.gz / *.tgz archives")
    ap.add_argument("--dest", default=None,
                    help="extraction root; each archive is extracted to "
                         "<dest>/<archive path relative to --src-root>/, preserving the "
                         "archive-tree hierarchy (default: extract in place, next to the "
                         "archive)")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the dataset dirs already exist")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going after a bad archive; failed archives are listed at the end")
    ap.add_argument("--workers", type=int, default=8,
                    help="archives extracted in parallel (1 = sequential; default: 8)")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable progress/ETA lines")
    ap.add_argument("--progress-interval", type=float, default=10.0,
                    help="seconds between progress lines (default: 10)")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    if not src_root.is_dir():
        log(f"src-root not found: {src_root}")
        sys.exit(1)
    log(f"EXTRACT_SCAN scanning {src_root} recursively for archives ...")
    archives = find_archives(src_root)
    if not archives:
        log(f"no *.tar.gz / *.tgz archives found under {src_root}")
        sys.exit(1)

    prog = Progress(len(archives), "archives", "extract",
                    enabled=not args.no_progress,
                    interval=args.progress_interval)
    prog.start_heartbeat()
    log(f"EXTRACT_START archives={len(archives)} src={src_root}")
    n_ok = n_skip = 0
    failed: list[str] = []
    workers = max(1, int(args.workers))
    dest_opt = Path(args.dest) if args.dest else None

    if workers == 1:
        for a in archives:
            dest_root = dest_for(a, src_root, dest_opt)
            prog.set_note(f"extracting: {a.name}")
            t0 = time.time()
            try:
                r = extract_archive(a, dest_root, args.force)
            except Exception as e:  # noqa: BLE001
                log(f"[FAIL] {a}: {type(e).__name__}: {e}")
                failed.append(str(a))
                prog.update(1, note=a.name)
                if not args.continue_on_error:
                    break
                continue
            if r == "skip":
                n_skip += 1
                log(f"[SKIP] {a}: already extracted (use --force to redo)")
            else:
                n_ok += 1
                size_mb = a.stat().st_size / 1e6
                log(f"[OK]   {a}: {size_mb:.0f} MB archive -> {dest_root} "
                    f"({time.time() - t0:.1f}s)")
            prog.update(1, note=a.name)
    else:
        # bounded submission: keep at most `workers` archives in flight, so the
        # progress note can show what is CURRENTLY being extracted
        pending = [(a, dest_for(a, src_root, dest_opt)) for a in archives]
        inflight: dict = {}
        cancelled = False

        def submit_one(ex) -> None:
            a, dest = pending.pop(0)
            inflight[ex.submit(_extract_worker, str(a), str(dest), args.force)] = (a, dest)
            names = [x[0].name for x in inflight.values()]
            shown = ", ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else "")
            prog.set_note(f"extracting: {shown}")

        with ProcessPoolExecutor(max_workers=workers) as ex:
            while pending and len(inflight) < workers:
                submit_one(ex)
            while inflight:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    a, dest_root = inflight.pop(fut)
                    if cancelled:
                        continue
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa: BLE001  (worker crash/pickling)
                        res = {"ok": False, "archive": str(a),
                               "error": f"{type(e).__name__}: {e}", "seconds": 0.0}
                    if res["ok"]:
                        if res["status"] == "skip":
                            n_skip += 1
                            log(f"[SKIP] {a}: already extracted (use --force to redo)")
                        else:
                            n_ok += 1
                            log(f"[OK]   {a}: {res['size_mb']:.0f} MB archive -> {dest_root} "
                                f"({res['seconds']:.1f}s)")
                        prog.update(1, note=a.name)
                    else:
                        log(f"[FAIL] {a}: {res['error']}")
                        failed.append(str(a))
                        if not args.continue_on_error:
                            cancelled = True
                            pending.clear()
                            for f2 in inflight:
                                f2.cancel()
                if not cancelled:
                    while pending and len(inflight) < workers:
                        submit_one(ex)

    prog.finish(ok=not failed)
    log(f"EXTRACT_DONE ok={n_ok} skip={n_skip} failed={len(failed)}")
    if failed:
        log("FAILED_ARCHIVES:")
        for f in failed:
            log(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
