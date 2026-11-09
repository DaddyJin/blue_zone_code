"""Bounded parallel native v2.1 packet-copy merge and full decode; no encoder."""
import os,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import media
_ORIGINAL_DECODE=media.decode_timing

def workers():return int(os.environ.get('GM100_VIDEO_WORKERS','4'))

def prepare(config,episodes,base,output):
    start=time.monotonic()
    if config['media']['policy']!='native':raise ValueError('GM100 backend accepts native v2.1 only')
    cameras=set(episodes[0]['videos']);paths=[]
    for e in episodes:
        if set(e['videos'])!=cameras:raise ValueError('Camera set changes inside task')
        for cam,v in e['videos'].items():
            if any(s in cam for s in ('/','\\','..')):raise ValueError('Unsafe camera name')
            if any(k in v for k in ('from_timestamp','from_frame')):raise ValueError('Expected complete v2.1 episode video')
            paths.append((base/v['path']).resolve())
    with ThreadPoolExecutor(max_workers=workers()) as pool:headers=dict(zip(paths,pool.map(media.probe,paths)))
    groups=defaultdict(list);profiles={};lengths={};maps={}
    for ep,e in enumerate(episodes):
        lengths[ep]=int(e['length']);maps[ep]={}
        for cam,v in e['videos'].items():
            path=(base/v['path']).resolve();h=headers[path]
            if h['frames']!=lengths[ep] or h['fps'] is None or abs(h['fps']-float(config['fps']))>1e-5:raise ValueError('Video count/fps disagrees with source rows: '+str(path))
            sig=(h['width'],h['height'],h['codec_name'],h['fps'])
            if cam in profiles and profiles[cam]!=sig:raise ValueError('Mixed format within camera; explicitly split task')
            profiles[cam]=sig
            groups[(cam,tuple(h['signature']))].append((path,{'episode_index':ep,'length':lengths[ep]},h))
    jobs=[];indices=defaultdict(int)
    for (cam,sig),items in groups.items():
        batch=[];size=0
        for item in items:
            b=item[0].stat().st_size
            if batch and size+b>config['media'].get('max_file_bytes',128_000_000):jobs.append((cam,indices[cam],batch));indices[cam]+=1;batch=[];size=0
            batch.append(item);size+=b
        if batch:jobs.append((cam,indices[cam],batch));indices[cam]+=1
    def merge(job):
        cam,i,batch=job;rel=f'videos/observation.images.{cam}/chunk-{i//1000:03d}/file-{i%1000:03d}.mp4';dest=output/rel
        if len(batch)==1:
            method=media._reuse(batch[0][0],dest,os.environ.get('GM100_LINK_MODE','copy'));offsets={batch[0][1]['episode_index']:0}
        else:
            rows=media._concat_group(batch,dest,'observation.images.'+cam,i)
            if any(r['invalid'] for r in rows.values()):raise ValueError('Packet count disagrees with frame count')
            offsets={ep:r['video_frame_offset'] for ep,r in rows.items()};method='packet_copy_concat'
        mapping={}
        for path,row,h in batch:
            mapping[row['episode_index']]=dict(file=rel,video_frame_offset=offsets[row['episode_index']],length=row['length'],chunk_index=i//1000,file_index=i%1000,source=str(path),source_from_frame=0,source_fps=h['fps'],image_transform=np.eye(3).tolist())
        return cam,mapping,dict(cam=cam,path=rel,method=method,sources=[str(x[0]) for x in batch],header=media.probe(dest),reencoded=False)
    files=[]
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        for cam,mapping,f in pool.map(merge,jobs):
            files.append(f)
            for ep,v in mapping.items():maps[ep][cam]=v
    return dict(episodes=maps,files=files,lengths=lengths,policy='native',prepare_seconds=time.monotonic()-start)

def decode_timing(output,m):
    def one(f):return _ORIGINAL_DECODE(output,{'files':[f],'episodes':{}})
    cache={};reports=[]
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        for c,r in pool.map(one,m['files']):cache.update(c);reports.extend(r)
    for ep,cameras in m['episodes'].items():
        reference=None
        for cam,v in cameras.items():
            axis,path=cache[v['file']];start=v['video_frame_offset'];end=start+v['length']
            if end>len(axis):raise ValueError('Episode beyond decoded boundary')
            local=axis[start:end]-axis[start]
            if reference is not None and not np.allclose(local,reference,atol=1e-6,rtol=0):raise ValueError('Camera presentation timelines differ')
            if not np.allclose(local,np.arange(v['length'])/v['source_fps'],atol=1e-5,rtol=0):raise ValueError('Unexpected variable frame timing')
            reference=local
            stop=float(axis[end]) if end<len(axis) else float(axis[-1]+1/v['source_fps'])
            v.update(from_timestamp=float(axis[start]),to_timestamp=stop,pts_path=path)
    return cache,reports

def install():media.prepare=prepare;media.decode_timing=decode_timing
