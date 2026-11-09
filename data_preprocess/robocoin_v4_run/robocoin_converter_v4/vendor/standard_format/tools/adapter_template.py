"""Copy this file; implement only source discovery/extraction and field choices.

The generic converter writes v3 metadata, vocabularies, video concat and 55D.
A data reference may be a per-episode parquet, or a slice of a shared v3 file:
  {"path":"source/data/chunk-000/file-000.parquet","offset":100,"length":20}
or an explicit source episode filter:
  {"path":"source/data/...parquet","episode_index":7}
Multiple references are concatenated in given order. Video intervals use
{"path":"source/videos/...mp4","from_frame":100,"length":20,"fps":10}.
Do not infer robot ownership, calibration, clock quality or action semantics.
"""
import json
from pathlib import Path

def build_manifest(source_root):
    # 1. Return one delivery unit: one robot + one semantic profile + one split.
    # 2. Read true source timestamps/instructions; pre-align state/action streams.
    # 3. Start with examples/robot.json, ego_mixed.json, or video_only.json.
    # 4. Use slot names, explicit quantity/unit/provenance/owner and frame ids.
    # 5. Preserve group identity when slicing one original video/session.
    raise NotImplementedError('Implement your dataset adapter here; mapping choices must be explicit')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source_root');p.add_argument('output');a=p.parse_args()
    Path(a.output).write_text(json.dumps(build_manifest(Path(a.source_root)),ensure_ascii=False,indent=2)+'\n')
