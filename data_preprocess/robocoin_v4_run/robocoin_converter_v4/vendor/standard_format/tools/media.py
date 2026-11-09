"""Native MP4 reuse / packet-copy merge, or explicit frame-preserving normalization."""
from __future__ import annotations
import hashlib,os,shutil,uuid
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import av
import numpy as np
from common import dump

def _header(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        first_packet = next((packet for packet in container.demux(stream) if packet.size), None)
        if first_packet is None or first_packet.pts is None or first_packet.dts is None:
            raise ValueError(f"Empty or timestamp-less video: {path}")
        decode_lead = first_packet.pts - first_packet.dts
        # Equality is intentionally conservative: differently encoded source
        # streams are put in separate output files rather than re-encoded.
        signature = (codec.name, codec.width, codec.height, str(codec.pix_fmt),
                     str(stream.time_base), str(stream.average_rate), codec.has_b_frames,
                     hashlib.sha256(codec.extradata or b"").hexdigest(), decode_lead)
        return {"signature": signature, "fps": float(stream.average_rate) if stream.average_rate else None,
                "time_base": str(stream.time_base), "width": codec.width, "height": codec.height,
                "codec": codec.name, "pix_fmt": codec.pix_fmt,
                "first_packet_keyframe": bool(first_packet.is_keyframe), "decode_lead_ticks": decode_lead,
                "frames": stream.frames, "duration": float(stream.duration * stream.time_base) if stream.duration is not None else None,
                "start_time": float((stream.start_time or 0) * stream.time_base)}

def _reuse(source, destination, mode="auto"):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return "hardlink_existing"
        raise FileExistsError(f"Refusing to overwrite existing media {destination}")
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    try:
        try:
            if mode == "copy":
                raise OSError("copy requested")
            os.link(source, temporary)
            method = "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
            shutil.copyfile(source, temporary)
            method = "copy"
        os.replace(temporary, destination)
        return method
    finally:
        temporary.unlink(missing_ok=True)

def _mapping(row, key, chunk, file_index, start, end, frame_count, invalid=False, reason=None):
    return {"source_episode_index": int(row["episode_index"]), "episode_index": int(row["episode_index"]),
            "chunk_index": int(chunk), "file_index": int(file_index),
            "video_chunk_index": int(chunk), "video_file_index": int(file_index),
            "from_timestamp": float(start), "to_timestamp": float(end),
            "source_video_key": key, "source_frame_count": int(frame_count), "frame_count": int(frame_count),
            "invalid": bool(invalid), "reason": reason}

def _concat_group(items, destination, key, file_index):
    """Copy packets, preserve presentation order and B-frame decode offsets.

    Each input must contain complete independently decodable GOPs, as the
    episode files do. A decode-time collision fails closed, never silently
    fabricates timestamps. The temporary output is discarded on failure.
    """
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    mappings, offset, previous_dts = {}, 0, None
    output = None
    try:
        for path, row, header in items:
            with av.open(str(path)) as source:
                stream = source.streams.video[0]
                tb = stream.time_base
                if output is None:
                    output = av.open(str(temporary), "w", format="mp4")
                    target = output.add_stream_from_template(stream, opaque=True)
                    target.time_base = tb
                first_pts, count, end, first_key = None, 0, None, None
                for packet in source.demux(stream):
                    if packet.size == 0:
                        continue
                    if packet.pts is None or packet.dts is None:
                        raise ValueError(f"Missing packet timestamps: {path}")
                    if first_pts is None:
                        first_pts = stream.start_time if stream.start_time is not None else packet.pts
                        first_key = packet.is_keyframe
                        if not first_key:
                            raise ValueError(f"Episode does not begin with keyframe: {path}")
                    duration = packet.duration
                    if not duration or duration <= 0:
                        raise ValueError(f"Missing positive packet duration: {path}")
                    raw_end = packet.pts - first_pts + duration
                    end = max(end or 0, raw_end)
                    packet.pts = packet.pts - first_pts + offset
                    packet.dts = packet.dts - first_pts + offset
                    if previous_dts is not None and packet.dts <= previous_dts:
                        raise ValueError(f"Nonmonotonic decode time at episode {row['episode_index']}: {path}")
                    previous_dts = packet.dts
                    packet.stream = target
                    output.mux(packet)
                    count += 1
                if not count:
                    raise ValueError(f"Empty source video: {path}")
                expected = int(row.get("length", count))
                mismatch = count != expected
                mapping = _mapping(row, key, 0, file_index, offset * tb, (offset + end) * tb, count,
                                   mismatch, f"video_frame_count_{count}_vs_state_{expected}" if mismatch else None)
                mapping["source_video_path"] = str(path)
                mapping["source_pts_origin"] = first_pts
                mapping["source_time_base"] = str(tb)
                mapping["video_frame_offset"] = sum(x["frame_count"] for x in mappings.values())
                mappings[int(row["episode_index"])] = mapping
                offset += end
        output.close()
        output = None
        os.replace(temporary, destination)
        return mappings
    finally:
        if output is not None:
            output.close()
        temporary.unlink(missing_ok=True)


def probe(path):
    h=_header(path)
    if not h['frames']:
        with av.open(str(path)) as c:h['frames']=sum(1 for _ in c.decode(video=0))
    h['codec_name']={'libdav1d':'av1','libaom-av1':'av1'}.get(h['codec'],h['codec'])
    return h

def geometry(sw,sh,tw,th):
    r=min(tw/sw,th/sh);w=max(1,round(sw*r));h=max(1,round(sh*r));x=(tw-w)//2;y=(th-h)//2
    return w,h,x,y,[[w/sw,0,x],[0,h/sh,y],[0,0,1]]

def normalize_clip(source,target,start,length,fps,size):
    """One input frame -> one output frame, no fps filter / duplication / dropping."""
    tw,th=map(int,size)
    if tw%2 or th%2 or min(tw,th)<=0:raise ValueError('H264 yuv420p target dimensions must be positive even numbers')
    rate=Fraction(str(fps)).limit_denominator(100000)
    target.parent.mkdir(parents=True,exist_ok=True);count=0;affine=None
    with av.open(str(source)) as inp,av.open(str(target),'w',format='mp4') as out:
        ins=inp.streams.video[0];ins.codec_context.thread_count=1
        enc=out.add_stream('libx264',rate=rate);enc.width=tw;enc.height=th;enc.pix_fmt='yuv420p'
        enc.codec_context.thread_count=1;enc.options={'crf':'18','preset':'fast','bf':'0'}
        for i,frame in enumerate(inp.decode(ins)):
            if i<start:continue
            if count>=length:break
            w,h,x,y,affine=geometry(frame.width,frame.height,tw,th)
            image=np.zeros((th,tw,3),np.uint8);image[y:y+h,x:x+w]=frame.reformat(width=w,height=h,format='rgb24').to_ndarray()
            f=av.VideoFrame.from_ndarray(image,format='rgb24');f.pts=count;f.time_base=Fraction(rate.denominator,rate.numerator)
            for packet in enc.encode(f):out.mux(packet)
            count+=1
        for packet in enc.encode():out.mux(packet)
    if count!=length:raise ValueError(f'Video ended early: {source}, decoded {count}, expected {length}')
    return affine


def packet_axis(path, expected):
    # Fast container-axis inspection for existing v3 intervals; full decode still follows.
    pts=[]
    with av.open(str(path)) as c:
        stream=c.streams.video[0]
        for packet in c.demux(stream):
            if packet.size:
                if packet.pts is None:raise ValueError('Packet without PTS')
                pts.append(float(packet.pts*packet.time_base))
    a=np.sort(np.asarray(pts,np.float64))
    if len(a)!=expected or not np.isfinite(a).all() or np.any(np.diff(a)<=0):raise ValueError('Cannot establish one packet / frame ordinal; use a dataset-specific adapter')
    return a

def prepare(config, episodes, base, output):
    policy=config['media']['policy'];nominal=float(config['fps']);max_bytes=int(config['media'].get('max_file_bytes',500_000_000))
    if policy not in ('native','normalize'):raise ValueError('media.policy must explicitly be native or normalize')
    if policy=='normalize' and float(config['media']['fps'])!=nominal:raise ValueError('media.fps must equal dataset fps after normalization')
    clips=[];headers={};profiles={};occurrences=defaultdict(int);source_axes={}
    cameras=set(episodes[0]['videos'])
    if not cameras:
        lengths={i:int(e['length']) for i,e in enumerate(episodes)}
        if any(n<=0 for n in lengths.values()) or any(e['videos'] for e in episodes):raise ValueError('Data-only episodes require positive length and no mixed video presence')
        return {'episodes':{i:{} for i in lengths},'files':[],'lengths':lengths,'policy':policy}
    for ep,e in enumerate(episodes):
        if set(e['videos'])!=cameras:raise ValueError('Camera keys differ between episodes; split unit or align in adapter')
        for cam,source in e['videos'].items():
            if not cam or any(x in cam for x in ['/', '\\', '..']):raise ValueError('Unsafe camera name')
            spec={'path':source} if isinstance(source,str) else source
            p=(base/spec['path']).resolve();h=headers.setdefault(str(p),None)
            if h is None:h=probe(p);headers[str(p)]=h
            if 'from_timestamp' in spec:
                if str(p) not in source_axes:source_axes[str(p)]=packet_axis(p,h['frames'])
                axis=source_axes[str(p)];target=float(spec['from_timestamp']);tolerance=float(spec.get('timestamp_tolerance',1e-4))
                first=int(np.argmin(np.abs(axis-target)))
                if abs(axis[first]-target)>tolerance:raise ValueError('Existing episode timestamp does not match actual PTS; review source timing')
            else:first=int(spec.get('from_frame',0))
            length=int(spec.get('length',h['frames']-first))
            if 'to_timestamp' in spec:
                axis=source_axes[str(p)];end=int(np.searchsorted(axis,float(spec['to_timestamp'])-float(spec.get('timestamp_tolerance',1e-4))))
                if end-first!=length:raise ValueError('Existing episode time interval / row count mismatch')
            if first<0 or length<=0 or first+length>h['frames']:raise ValueError('Invalid source video frame range')
            declared=float(spec.get('fps',e.get('fps',h['fps'])))
            if policy=='native':
                signature=(h['width'],h['height'],h['codec_name'])
                if cam in profiles and profiles[cam]!=signature:raise ValueError(f'{cam}: mixed geometry/codec; select normalize or split delivery unit')
                profiles[cam]=signature
                if abs(declared-nominal)>max(1e-6,nominal*1e-5):raise ValueError('Mixed episode fps; select normalize (frame-preserving retime) or split unit')
            clips.append(dict(ep=ep,cam=cam,path=p,first=first,length=length,header=h,source_fps=declared,affine=np.eye(3).tolist()))
            occurrences[(cam,str(p))]+=1
    lengths={}
    for c in clips:
        if c['ep'] in lengths and lengths[c['ep']]!=c['length']:raise ValueError('One row per camera frame required; align asynchronous cameras in adapter')
        lengths[c['ep']]=c['length']
    mappings=defaultdict(dict);files=[];groups=defaultdict(list);indices=defaultdict(int);reuse={}
    def target(cam,index):
        chunk,fi=divmod(index,1000)
        return output/'videos'/('observation.images.'+cam)/f'chunk-{chunk:03d}'/f'file-{fi:03d}.mp4'
    for c in clips:
        if policy=='normalize':
            p=output/'.work'/c['cam']/f"episode-{c['ep']:06d}.mp4"
            size=config['media']['size'];c['affine']=normalize_clip(c['path'],p,c['first'],c['length'],nominal,size)
            h=probe(p);groups[(c['cam'],tuple(h['signature']))].append((p,{'episode_index':c['ep'],'length':c['length']},h,c))
        elif occurrences[(c['cam'],str(c['path']))]==1 and c['first']==0 and c['length']==c['header']['frames']:
            groups[(c['cam'],tuple(c['header']['signature']))].append((c['path'],{'episode_index':c['ep'],'length':c['length']},c['header'],c))
        else:
            key=(c['cam'],str(c['path']))
            if key not in reuse:
                i=indices[c['cam']];indices[c['cam']]+=1;p=target(c['cam'],i)
                method=_reuse(c['path'],p,'auto');reuse[key]=(p,i)
                files.append(dict(cam=c['cam'],path=str(p.relative_to(output)),method=method,sources=[str(c['path'])],header=c['header']))
            p,i=reuse[key]
            mappings[c['ep']][c['cam']]=dict(file=str(p.relative_to(output)),video_frame_offset=c['first'],length=c['length'],chunk_index=i//1000,file_index=i%1000,source=str(c['path']),source_from_frame=c['first'],source_fps=c['source_fps'],image_transform=c['affine'])
    for (cam,_),items in groups.items():
        batches=[];batch=[];size=0
        for item in items:
            b=item[0].stat().st_size
            if batch and size+b>max_bytes:batches.append(batch);batch=[];size=0
            batch.append(item);size+=b
        if batch:batches.append(batch)
        for batch in batches:
            i=indices[cam];indices[cam]+=1;p=target(cam,i);p.parent.mkdir(parents=True,exist_ok=True)
            if len(batch)==1:
                method=_reuse(batch[0][0],p,'auto');offsets={batch[0][1]['episode_index']:0}
            else:
                rows=_concat_group([x[:3] for x in batch],p,'observation.images.'+cam,i)
                if any(r['invalid'] for r in rows.values()):raise ValueError('Source packet/frame mismatch; quarantine explicitly in adapter')
                offsets={ep:r['video_frame_offset'] for ep,r in rows.items()};method='packet_copy_concat'
            for _,row,_,c in batch:
                mappings[row['episode_index']][cam]=dict(file=str(p.relative_to(output)),video_frame_offset=offsets[row['episode_index']],length=c['length'],chunk_index=i//1000,file_index=i%1000,source=str(c['path']),source_from_frame=c['first'],source_fps=c['source_fps'],image_transform=c['affine'])
            files.append(dict(cam=cam,path=str(p.relative_to(output)),method=method,sources=[str(x[3]['path']) for x in batch],header=probe(p),reencoded=(policy=='normalize')))
    return {'episodes':dict(mappings),'files':files,'lengths':lengths,'policy':policy}

def decode_timing(output, media):
    """Full decode: exact PTS, finite strictly increasing, no silent corruption skip."""
    cache={};reports=[]
    for f in media['files']:
        p=output/f['path'];pts=[];sizes=set()
        with av.open(str(p)) as c:
            s=c.streams.video[0];s.codec_context.thread_count=int(os.environ.get("ROBOCOIN_DECODE_THREADS","1"));s.thread_type="AUTO"
            for frame in c.decode(s):
                if getattr(frame,'is_corrupt',False):raise ValueError('Decoder marked corrupt frame: '+str(p))
                if frame.pts is None:raise ValueError('Frame has no actual PTS: '+str(p))
                pts.append(float(frame.pts*frame.time_base));sizes.add((frame.width,frame.height))
        a=np.asarray(pts,np.float64)
        if len(a)!=f['header']['frames'] or not np.isfinite(a).all() or np.any(np.diff(a)<=0):raise ValueError('Decoded count / actual PTS invalid: '+str(p))
        if sizes!={(f['header']['width'],f['header']['height'])}:raise ValueError('Dynamic resolution requires explicit adapter')
        relative=Path(f['path']).relative_to('videos').with_suffix('.pts.npy');q=output/'meta/video_timing'/relative;q.parent.mkdir(parents=True,exist_ok=True);np.save(q,a,allow_pickle=False)
        cache[f['path']]=(a,str(q.relative_to(output)))
        reports.append({'file':f['path'],'frames':len(a),'passed':True})
    for ep,cameras in media['episodes'].items():
        for cam,m in cameras.items():
            a,path=cache[m['file']];first=m['video_frame_offset'];end=first+m['length']
            if end>len(a):raise ValueError('Episode beyond decoded video boundary')
            step=(a[-1]-a[-2]) if len(a)>1 else 1/float(next(f['header']['fps'] for f in media['files'] if f['path']==m['file']))
            m.update(from_timestamp=float(a[first]),to_timestamp=float(a[end]) if end<len(a) else float(a[-1]+step),pts_path=path)
    return cache,reports
