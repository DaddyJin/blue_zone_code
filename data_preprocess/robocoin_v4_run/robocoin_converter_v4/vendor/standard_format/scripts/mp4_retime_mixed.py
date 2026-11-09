#!/usr/bin/env python
"""Retime a "mixed"-type merged mp4 onto a uniform CFR30 timeline.

Background: round-1 egostandard merged episode videos were packed at their
native frame rates (25fps etc. at mdhd timescale 12800) while the data side
assumes 30fps.  concat-demuxer merging of inputs with heterogeneous time
bases additionally produced broken segments (giant stts deltas, segments at
delta 1000 ticks).  Packet *bytes* are intact; only container timing is bad.

Strategy (lossless container rewrite, packets copied bit-for-bit):
  decode order  = file order          -> dts[k] = k * 512 @ timescale 15360
  presentation  = rank by old pts     -> pts    = rank * 512 @ 15360
  duration      = 512

Works because the bogus gap deltas are always positive, so sorting by old
pts yields segment1, then segment2, ... in true presentation order.

Usage: mp4_retime_mixed.py <in.mp4> <out.mp4> [--mode pts|poc]
  pts: presentation order = sort by container pts (healthy/mixed files)
  poc: presentation order = H.264 POC from bitstream (garbage files where
       container pts were forged by concat; ground truth from slice headers)
Writes to tmp then renames. Idempotent via caller-side .retime_done marker.
"""
import os
import sys
from fractions import Fraction

import av

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h264_poc import compute_pocs

TICK = 512          # 1/30 s at timescale 15360
TIMESCALE = 15360


def main() -> int:
    src_path, dst_path = sys.argv[1], sys.argv[2]
    mode = "pts"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    assert mode in ("pts", "poc")

    inp = av.open(src_path, "r")
    vstreams = [s for s in inp.streams if s.type == "video"]
    assert len(vstreams) == 1, f"expected 1 video stream, got {len(vstreams)}"
    assert all(s.type != "audio" for s in inp.streams), "audio stream present"
    vs = vstreams[0]

    # pass 1: collect packets in file (decode) order, keeping the objects
    # (their buffers stay refcounted; input container can then close)
    pkts = [p for p in inp.demux(vs) if p.dts is not None]
    old_pts = [p.pts if p.pts is not None else p.dts for p in pkts]

    # output stream must be created from the template while input is open
    out = av.open(dst_path + ".tmp", "w", format="mp4")
    ovs = out.add_stream_from_template(vs)
    ovs.time_base = Fraction(1, TIMESCALE)
    extradata = vs.codec_context.extradata
    inp.close()

    n = len(pkts)
    assert n > 0, "no packets"

    # presentation rank
    if mode == "pts":
        order = sorted(range(n), key=lambda i: (old_pts[i], i))
    else:
        keys = compute_pocs([bytes(p) for p in pkts], extradata)
        order = sorted(range(n), key=lambda i: (keys[i], i))
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r
    # B-frame reorder depth: shift dts into the negative so pts >= dts
    # (same shape as ffmpeg-produced files with edit-list start offset)
    reorder = max(i - rank[i] for i in range(n))

    tb = Fraction(1, TIMESCALE)
    for i, p in enumerate(pkts):
        p.stream = ovs
        p.dts = (i - reorder) * TICK
        p.pts = rank[i] * TICK
        p.duration = TICK
        p.time_base = tb
        out.mux(p)
    out.close()
    os.replace(dst_path + ".tmp", dst_path)
    print(f"retimed {n} packets -> {dst_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
