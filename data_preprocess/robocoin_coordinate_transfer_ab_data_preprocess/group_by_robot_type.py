#!/usr/bin/env python3
"""Group subdatasets by canonical mechanical-arm family using absolute symlinks.

Input P:
    P/
      dataset_a/
        meta/info.json
      dataset_b/
        meta/info.json
      ...

Output Q:
    Q/
      Unitree_G1/
        dataset_a -> /absolute/path/P/dataset_a
      Agilex_Cobot_Magic/
        dataset_b -> /absolute/path/P/dataset_b
      unknown/
        ...
      manifest.json

The group is decided by the dataset prefix (the physical arm family), because
the published robot_type is inconsistent (e.g. Cobot Magic can be labelled
``Agilex_Cobot_Magic`` or ``aloha``).  Raw robot_type is retained in
manifest.json for traceability and used only as a fallback.

Only direct children of P containing meta/info.json are treated as datasets.
No data is copied.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GENERATOR = "group_by_robot_type"
UNKNOWN = "unknown"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_valid_robot_type(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if text.startswith("publisher_robot_type_unspecified"):
        return False
    return True


def robot_type_for(dataset_dir: Path) -> Tuple[Optional[str], str]:
    """Return (raw robot_type, source).

    Prefer meta/info.json. If it only contains the publisher placeholder, fall
    back to meta/input_manifest.json, which data-cleaning often fills with a
    real robot_type.
    """
    info = read_json(dataset_dir / "meta" / "info.json") or {}
    value = info.get("robot_type")
    if is_valid_robot_type(value):
        return str(value).strip(), "meta/info.json"

    manifest = read_json(dataset_dir / "meta" / "input_manifest.json") or {}
    value = manifest.get("robot_type")
    if is_valid_robot_type(value):
        return str(value).strip(), "meta/input_manifest.json"

    return None, "unknown"


def sanitize_group(raw: Optional[str]) -> str:
    if raw is None:
        return UNKNOWN
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or UNKNOWN


def canonical_group(dataset_name: str, raw_robot_type: Optional[str]) -> Tuple[str, str]:
    """Map a dataset to a mechanical-arm family.

    Published robot_type is inconsistent: for example Cobot Magic datasets can
    be labelled ``Agilex_Cobot_Magic`` or ``aloha``.  Group by the dataset
    prefix (the physical arm family) first, and keep the raw robot_type only as
    a fallback / provenance field.
    """
    name = dataset_name.lower()
    prefix_rules = [
        (("agilex_cobot_magic", "agilex_split_aloha"), "Agilex_Cobot_Magic", "dataset_prefix"),
        (("unitree", "g1edu"), "Unitree_G1", "dataset_prefix"),
        (("airbot",), "Airbot_MMK2", "dataset_prefix"),
        (("galaxea", "r1_lite"), "Galaxea_R1_Lite", "dataset_prefix"),
        (("leju_kuavo", "leju_robot"), "Leju_Kuavo_4", "dataset_prefix"),
        (("realman", "rmc-aida", "rmc_aida"), "Realman_RMC-AIDA-L", "dataset_prefix"),
        (("agibot",), "AgiBot_G1", "dataset_prefix"),
        (("ai2", "alphabot", "alpha_bot"), "AI2_AlphaBot_2", "dataset_prefix"),
        (("galbot",), "Galbot_G1", "dataset_prefix"),
        (("tianqin", "tianqing"), "Tianqing_A2", "dataset_prefix"),
    ]
    for prefixes, group, rule in prefix_rules:
        if any(name.startswith(prefix) for prefix in prefixes):
            return group, rule
    if raw_robot_type is not None:
        return sanitize_group(raw_robot_type), "robot_type_fallback"
    return UNKNOWN, "unrecognized"


def find_datasets(root: Path) -> List[Path]:
    """Return direct child datasets (or root itself if it is one dataset)."""
    if (root / "meta" / "info.json").is_file():
        return [root]
    datasets = []
    if not root.is_dir():
        return datasets
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "meta" / "info.json").is_file():
            datasets.append(child)
    return datasets


def _remove_previous_output(output: Path, source: Path) -> None:
    """Safely replace a previous Q created by this script.

    We only remove symlinks inside group directories listed in the previous
    manifest. Any real file/dir is treated as unexpected and aborts, so a wrong
    output path cannot silently delete user data.
    """
    if not output.exists():
        return
    if not output.is_dir():
        raise RuntimeError(f"output exists but is not a directory: {output}")
    entries = list(output.iterdir())
    if not entries:
        return

    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if not manifest or manifest.get("generator") != GENERATOR:
        raise RuntimeError(
            f"output is non-empty and was not created by {GENERATOR}: {output}\n"
            "remove it manually or choose a different output path"
        )
    if str(manifest.get("source")) != str(source):
        raise RuntimeError(
            f"previous output was created from {manifest.get('source')}, "
            f"not {source}; remove it manually or choose a different output path"
        )

    for group, items in (manifest.get("groups") or {}).items():
        group_dir = output / group
        if not group_dir.exists() and not group_dir.is_symlink():
            continue
        if group_dir.is_symlink() or not group_dir.is_dir():
            raise RuntimeError(f"unexpected non-directory in output: {group_dir}")
        for child in group_dir.iterdir():
            if child.is_symlink():
                child.unlink()
            else:
                raise RuntimeError(
                    f"unexpected non-symlink in managed group: {child}"
                )
        group_dir.rmdir()

    manifest_path.unlink(missing_ok=True)


def build_groups(datasets: Iterable[Path]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for dataset in datasets:
        raw, source = robot_type_for(dataset)
        group, rule = canonical_group(dataset.name, raw)
        groups.setdefault(group, []).append(
            {
                "name": dataset.name,
                "path": str(dataset),
                "target": str(dataset.resolve()),
                "robot_type_raw": raw,
                "robot_type_source": source,
                "group_rule": rule,
            }
        )
    return groups


def create_links(output: Path, groups: Dict[str, List[Dict[str, Any]]]) -> None:
    for group, items in sorted(groups.items()):
        group_dir = output / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            link = group_dir / item["name"]
            if link.exists() or link.is_symlink():
                # The safe cleanup should have removed previous links. A
                # collision here means duplicate dataset names or a manual
                # intervention; replace only if it is already a symlink.
                if link.is_symlink():
                    link.unlink()
                else:
                    raise RuntimeError(f"refusing to replace non-symlink: {link}")
            os.symlink(item["target"], link)


def write_manifest(
    output: Path,
    source: Path,
    groups: Dict[str, List[Dict[str, Any]]],
) -> None:
    payload = {
        "generator": GENERATOR,
        "source": str(source),
        "output": str(output),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "link_mode": "absolute",
        "total_datasets": sum(len(v) for v in groups.values()),
        "groups": groups,
    }
    path = output / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Group P's subdatasets into Q/<robot_type>/<dataset> symlinks."
    )
    parser.add_argument("--input", required=True, help="parent directory P")
    parser.add_argument("--output", required=True, help="output directory Q")
    args = parser.parse_args(argv)

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.is_dir():
        print(f"error: input not found: {source}", file=sys.stderr)
        return 2
    datasets = find_datasets(source)
    if not datasets:
        print(
            f"error: no dataset found under {source} "
            "(expected direct children with meta/info.json)",
            file=sys.stderr,
        )
        return 2

    try:
        _remove_previous_output(output, source)
        output.mkdir(parents=True, exist_ok=True)
        groups = build_groups(datasets)
        create_links(output, groups)
        write_manifest(output, source, groups)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"source: {source}")
    print(f"output: {output}")
    print(f"datasets: {len(datasets)}  groups: {len(groups)}")
    for group, items in sorted(groups.items()):
        print(f"  {group}: {len(items)}")
    if UNKNOWN in groups:
        print(f"  note: {len(groups[UNKNOWN])} dataset(s) are under {UNKNOWN}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
