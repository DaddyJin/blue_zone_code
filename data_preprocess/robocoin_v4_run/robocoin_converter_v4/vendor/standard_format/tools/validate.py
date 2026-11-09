"""Independent structural/semantic QC; full decode optional; no source repairs."""
from __future__ import annotations
import argparse,json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from common import SLOTS,VECTORS,values

def validate(root,decode=False,samples=5):
    root=Path(root);errors=[];sampled=[];frames=0;episodes=[];tables=OrderedDict();seen_files={};intervals=[]
    try:
        info=json.loads((root/'meta/info.json').read_text());custom=info['custom']
        if info['codebase_version']!='v3.0' or custom['standard_format_version'] not in ('2.0','2.1-robocoin'):raise ValueError('Not Unified55 v2 / LeRobot v3')
        if float(info['fps'])<=0:errors.append('Invalid fps')
        for key in VECTORS:
            if info['features'].get(key,{}).get('shape')!=[55]:errors.append('Missing 55D feature '+key)
        for p in sorted((root/'meta/episodes').rglob('*.parquet')):episodes.extend(pq.ParquetFile(p).read().to_pylist())
        episodes.sort(key=lambda e:e['episode_index'])
        side={}
        with (root/'meta/episodes.jsonl').open() as f:
            for line in f:
                e=json.loads(line)
                if e['episode_index'] in side:errors.append('Duplicate sidecar episode')
                side[e['episode_index']]=e
        tasks=pq.ParquetFile(root/'meta/tasks.parquet').read();subtasks=pq.ParquetFile(root/'meta/subtasks.parquet').read()
        vocab={r['task_index']:r['task'] for r in tasks.to_pylist()};sub={r['subtask_index']:r['subtask'] for r in subtasks.to_pylist()}
        if list(sorted(vocab))!=list(range(len(vocab))) or list(sorted(sub))!=list(range(len(sub))):errors.append('Noncontiguous vocabulary ids')
        if len(vocab)!=info['total_tasks'] or not all(isinstance(x,str) for x in list(vocab.values())+list(sub.values())):errors.append('Invalid text vocabulary')
        for t,label in [(tasks,'task'),(subtasks,'subtask')]:
            md=t.schema.metadata or {};pandas=json.loads(md.get(b'pandas',b'{}'))
            if pandas.get('index_columns')!=[label]:errors.append('Vocabulary pandas index metadata missing '+label)
        if len(episodes)!=info['total_episodes'] or set(side)!=set(e['episode_index'] for e in episodes):errors.append('Episode metadata membership mismatch')
        if [e['episode_index'] for e in episodes]!=list(range(len(episodes))):errors.append('Noncontiguous episodes')
        expected_global=0;mode=custom['action']['mode']
        timingrows=pq.ParquetFile(root/'meta/video_timing/episodes.parquet').read().to_pylist();timing={(r['episode_index'],r['camera']):r for r in timingrows}
        for e in episodes:
            ep=e['episode_index'];n=e['length'];s=side[ep]
            if e['dataset_from_index']!=expected_global or e['dataset_to_index']!=expected_global+n:errors.append(f'EP {ep}: global row range')
            expected_global+=n;frames+=n
            p=root/info['data_path'].format(chunk_index=e['data/chunk_index'],file_index=e['data/file_index'])
            if p not in tables:
                tables[p]=pq.ParquetFile(p).read();seen_files[p]=len(tables[p])
                for name in tables[p].column_names:
                    if name not in info['features']:errors.append('Unregistered field '+name)
                if len(tables)>2:tables.popitem(last=False)
            tables.move_to_end(p)
            full=tables[p];a=e['data/file_from_offset'];b=e['data/file_to_offset'];t=full.slice(a,b-a)
            if len(t)!=n or not np.all(values(t['episode_index'])==ep) or not np.array_equal(values(t['frame_index']),np.arange(n)) or not np.array_equal(values(t['index']),np.arange(e['dataset_from_index'],e['dataset_to_index'])):errors.append(f'EP {ep}: row ownership')
            arrays={}
            for key in ('index','episode_index','frame_index','task_index','subtask_index'):
                if t.schema.field(key).type!=pa.int64():errors.append(f'EP {ep}: index dtype {key}')
            for key in ('mask_language','qc_sample_valid'):
                if t.schema.field(key).type!=pa.bool_():errors.append(f'EP {ep}: boolean dtype {key}')
            for key in VECTORS:
                typ=t.schema.field(key).type
                if not pa.types.is_fixed_size_list(typ) or typ.list_size!=55 or typ.value_type!=pa.float32():errors.append(f'EP {ep}: dtype {key}')
                x=values(t[key]);arrays[key]=x
                if x.shape!=(n,55) or not np.isfinite(x).all():errors.append(f'EP {ep}: nonfinite / shape {key}')
                if x[:,51:55].any():errors.append(f'EP {ep}: reserved slots not empty {key}')
            state=arrays['state_unified'];action=arrays['action_unified'];ms=arrays['mask_state'];ma=arrays['mask_action']
            if not np.isin(ms,[0,1]).all() or not np.isin(ma,[0,1]).all() or not np.array_equal(arrays['reliability'],ms):errors.append(f'EP {ep}: masks/reliability')
            declared_state=set();declared_action=set()
            for fields,declared in [(custom['state_fields'],declared_state),(custom['action'].get('fields',[]) if mode=='source' else custom['state_fields'] if mode=='next_state' else [],declared_action)]:
                for f in fields:
                    start,end=SLOTS[f['slot']]
                    components=f.get('components',list(range(len(f.get('indices',f.get('component_names',[]))))))
                    # Full EEF/gripper without indices infers actual enabled width from masks; reserved still forbidden below.
                    if not components:components=list(range(end-start))
                    declared.update(start+i for i in components)
            for mask,declared,label in [(ms,declared_state,'state'),(ma,declared_action,'action')]:
                forbidden=[i for i in range(55) if i not in declared]
                if forbidden and mask[:,forbidden].any():errors.append(f'EP {ep}: undeclared slots in {label}')
            if mode=='next_state':
                if ma[-1].any() or not np.allclose(action[:-1][ma[:-1].astype(bool)],state[1:][ma[:-1].astype(bool)],atol=1e-6):errors.append(f'EP {ep}: next-state action / terminal')
                if np.any(ma[:-1]>ms[1:]):errors.append(f'EP {ep}: action target has unavailable state component')
            if mode=='none' and (ma.any() or action.any()):errors.append(f'EP {ep}: absent action must be zero')
            if custom['data_kind']=='video_only' and any(x.any() for x in arrays.values()):errors.append(f'EP {ep}: video_only 55D not empty')
            for start,end in [(14,21),(21,28)]:
                for arr,mask in [(state,ms),(action,ma)]:
                    qm=mask[:,start+3:end]
                    if np.any(qm.any(1)!=qm.all(1)):errors.append(f'EP {ep}: partial quaternion mask')
                    good=qm.all(1)
                    if good.any() and not np.allclose(np.linalg.norm(arr[good,start+3:end],axis=1),1,atol=1e-4):errors.append(f'EP {ep}: quaternion normalization')
            for arr,mask,fields in [(state,ms,custom['state_fields']),(action,ma,custom['action'].get('fields',[]))]:
                for slot,name in [(28,'left_gripper'),(29,'right_gripper')]:
                    good=mask[:,slot]>0
                    rule=next((f.get('gripper',{}) for f in fields if f['slot']==name),{})
                    if rule.get('mode')=='native_open_direction':
                        if custom['standard_format_version']!='2.1-robocoin' or rule.get('scale') not in (-1,1) or not rule.get('direction_evidence'):errors.append(f'EP {ep}: undocumented native gripper')
                    elif np.any((arr[good,slot]<0)|(arr[good,slot]>1)):errors.append(f'EP {ep}: gripper range')
            if custom['standard_format_version']=='2.1-robocoin':
                sd=values(t['sample_dt']);sm=values(t['mask_sample_dt']).astype(bool);sk=values(t['sample_dt_kind'])
                if t.schema.field('sample_dt').type!=pa.float64() or not np.isfinite(sd).all() or np.any(sd[sm]<=0) or np.any(sd[~sm]!=0) or sm[-1] or sd[-1]!=0:errors.append(f'EP {ep}: invalid sample_dt')
                expected_kind=2 if s['source_time']['kind']=='measured' else 1
                if np.any(sk[sm]!=expected_kind) or np.any(sk[~sm]!=0):errors.append(f'EP {ep}: sample_dt provenance')
                base=values(t['source_timestamp']) if expected_kind==2 else values(t['source.timestamp']).astype(np.float64)
                if not np.allclose(sd[:-1],np.diff(base),atol=1e-9):errors.append(f'EP {ep}: sample interval mismatch')
            for key in ('timestamp','source_timestamp','source_dt','action_dt'):
                if t.schema.field(key).type!=pa.float64() or not np.isfinite(values(t[key])).all():errors.append(f'EP {ep}: time dtype/nonfinite {key}')
            if np.any(np.diff(values(t['timestamp']))<=0):errors.append(f'EP {ep}: nonmonotonic query timestamp; repair source clock in adapter')
            st=values(t['source_timestamp']);dt=values(t['source_dt']);valid_dt=values(t['mask_source_dt']).astype(bool)
            for key in ('mask_source_timestamp','mask_source_dt','mask_action_dt'):
                if t.schema.field(key).type!=pa.float32() or not np.isin(values(t[key]),[0,1]).all():errors.append(f'EP {ep}: invalid scalar mask {key}')
            source_valid=values(t['mask_source_timestamp']).astype(bool)
            if s['source_time']['kind']=='unknown' and (source_valid.any() or st.any()):errors.append(f'EP {ep}: unknown source clock marked valid')
            if np.any(dt[valid_dt]<=0) or np.any(valid_dt[:-1]&~(source_valid[:-1]&source_valid[1:])):errors.append(f'EP {ep}: invalid physical dt endpoints')
            if valid_dt[-1] or dt[-1]!=0:errors.append(f'EP {ep}: source_dt crosses episode')
            if s['source_time']['kind']!='measured' and (valid_dt.any() or dt.any()):errors.append(f'EP {ep}: guessed physical time')
            if valid_dt[:-1].any() and not np.allclose(dt[:-1][valid_dt[:-1]],np.diff(st)[valid_dt[:-1]],atol=1e-9):errors.append(f'EP {ep}: physical time interval mismatch')
            if np.any(dt[~valid_dt]!=0):errors.append(f'EP {ep}: invalid physical dt is not zero')
            adt=values(t['action_dt']);adtm=values(t['mask_action_dt']).astype(bool)
            if np.any(adt[~adtm]!=0) or np.any(adt[adtm]<=0):errors.append(f'EP {ep}: action_dt validity')
            if mode=='next_state' and (adtm[-1] or not np.allclose(adt[adtm],dt[adtm],atol=1e-9)):errors.append(f'EP {ep}: action horizon')
            ids=values(t['task_index']);sids=values(t['subtask_index'])
            if not np.isin(ids,list(vocab)).all() or not np.isin(sids,[-1]+list(sub)).all():errors.append(f'EP {ep}: unknown language id')
            else:
                expected_language=np.array([bool(vocab[int(i)].strip()) for i in ids])
                if not np.array_equal(values(t['mask_language']),expected_language):errors.append(f'EP {ep}: language mask')
                if set(s['tasks'])!={vocab[int(i)] for i in np.unique(ids)}:errors.append(f'EP {ep}: episode task text mismatch')
            expected_sub=np.full(n,-1,np.int64)
            for seg in s['action_config']:
                a0,b0=seg['start_frame'],seg['end_frame'];sid=next((i for i,text in sub.items() if text==seg['action_text']),None)
                if sid is None or not 0<=a0<b0<=n:errors.append(f'EP {ep}: subtask bounds/vocab');continue
                if np.any(expected_sub[a0:b0]!=-1):errors.append(f'EP {ep}: overlapping subtasks')
                expected_sub[a0:b0]=sid
            if not np.array_equal(sids,expected_sub):errors.append(f'EP {ep}: subtask/frame mismatch')
            if not s.get('video_valid',True) and (ms.any() or ma.any()):errors.append(f'EP {ep}: excluded video supervised')
            invalid=~values(t['qc_sample_valid']).astype(bool)
            if ms[invalid].any() or ma[invalid].any():errors.append(f'EP {ep}: invalid sample supervised')
            for key,f in info['features'].items():
                if f['dtype']!='video':continue
                cam=key.removeprefix('observation.images.');prefix='videos/'+key+'/';r=timing[(ep,cam)]
                path=root/info['video_path'].format(video_key=key,chunk_index=e[prefix+'chunk_index'],file_index=e[prefix+'file_index'])
                axis=np.load(root/r['pts_path'],mmap_mode='r',allow_pickle=False);first=r['video_frame_offset'];end=first+n
                if not path.is_file() or axis.dtype!=np.float64 or first<0 or end>len(axis) or not np.isfinite(axis).all() or np.any(np.diff(axis)<=0):errors.append(f'EP {ep}: video interval/PTS')
                if abs(float(axis[first])-e[prefix+'from_timestamp'])>1e-8:errors.append(f'EP {ep}: video from offset')
                if e[prefix+'to_timestamp']<=axis[end-1]:errors.append(f'EP {ep}: video to offset')
                intervals.append((str(path),r['pts_path']))
        if frames!=info['total_frames'] or sum(seen_files.values())!=frames:errors.append('Physical row total mismatch')
        for p,t in tables.items():
            for name in t.column_names:
                if name not in info['features']:errors.append('Unregistered field '+name)
        if decode:
            import av
            for p,axispath in set(intervals):
                actual=[]
                with av.open(p) as c:
                    stream=c.streams.video[0];stream.codec_context.thread_count=1
                    for frame in c.decode(stream):
                        if frame.pts is None:raise ValueError('Missing decoded PTS')
                        actual.append(float(frame.pts*frame.time_base))
                expected=np.load(root/axispath,allow_pickle=False)
                if not np.array_equal(np.asarray(actual,np.float64),expected):errors.append('Full decoded PTS differs '+p)
        if samples and not errors:
            from read_frame import StandardDataset
            ds=StandardDataset(root)
            for i in sorted(set(np.linspace(0,len(ds)-1,min(samples,len(ds)),dtype=int))) if len(ds) else []:
                row=ds[int(i)];sampled.append({'index':int(i),'episode_index':row['episode_index'],'frame_index':row['frame_index'],'cameras':{k:list(v.shape) for k,v in row.items() if k.startswith('observation.images.') and isinstance(v,np.ndarray)}})
            ds.close()
    except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
    return {'passed':not errors,'errors':errors,'frames':frames,'episodes':len(episodes),'sampled_reads':sampled,'full_decode':decode,'scope':'structural+55D+physical-time+vocabulary+subtasks+actual-PTS; source meaning comes from declared adapter mappings'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--decode',action='store_true');p.add_argument('--samples',type=int,default=8);p.add_argument('--output');a=p.parse_args();r=validate(a.root,a.decode,a.samples)
    if a.output:Path(a.output).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False));raise SystemExit(0 if r['passed'] else 1)
