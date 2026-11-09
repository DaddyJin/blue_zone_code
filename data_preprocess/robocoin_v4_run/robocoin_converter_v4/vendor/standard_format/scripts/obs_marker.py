#!/usr/bin/env python
"""生成并上传任务目录的"上传完成"标记到 OBS。

用途: 增量上传场景下, 下游同事无法判断某个任务文件夹是否传完。
约定: 任务所有数据文件上传成功后, 最后上传
  obs://<bucket>/<prefix>/<task>/_UPLOAD_COMPLETE.json
内容含文件清单(relpath+size)与总字节数, 下游下载后可逐项核对。
规则: 没有该标记的任务文件夹 = 未传完, 不要下载。

用法:
  obs_marker.py <task_dir> <bucket> <prefix>     # 生成+上传单个任务标记
凭证: 同 obs_batch_upload.py, 读 data_pipeline/.env
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MARKER_NAME = "_UPLOAD_COMPLETE.json"


def load_env() -> None:
    env = Path("/home/z00827204/data_pipeline/.env")
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def make_manifest(task_dir: Path) -> dict:
    files = []
    total = 0
    for dirpath, _dn, fn in os.walk(task_dir):
        for f in sorted(fn):
            if f == MARKER_NAME:
                continue
            p = Path(dirpath) / f
            rel = str(p.relative_to(task_dir))
            sz = p.stat().st_size
            files.append({"path": rel, "size": sz})
            total += sz
    return {
        "task": task_dir.name,
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
        "upload_finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "convention": "下载前必须看到本文件; 下载后按 files 清单逐个核对 size",
    }


def upload_marker(task_dir: Path, bucket: str, prefix: str) -> str:
    load_env()
    from obs import ObsClient
    manifest = make_manifest(task_dir)
    local = task_dir / MARKER_NAME
    local.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    key = f"{prefix.rstrip('/')}/{task_dir.name}/{MARKER_NAME}"
    c = ObsClient(access_key_id=os.environ["OBS_AK"],
                  secret_access_key=os.environ["OBS_SK"],
                  server=os.environ["OBS_ENDPOINT"])
    r = c.putFile(bucket, key, str(local))
    if r.status >= 300:
        raise RuntimeError(f"putFile {key} -> {r.status}: {r.body}")
    return f"obs://{bucket}/{key} ({manifest['file_count']} files, {manifest['total_bytes']} B)"


if __name__ == "__main__":
    tdir, bucket, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    print(upload_marker(tdir, bucket, prefix))
