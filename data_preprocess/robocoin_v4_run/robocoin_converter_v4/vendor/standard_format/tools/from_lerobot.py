"""Build source references for existing LeRobot v2.1/v3; semantics remain explicit."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

def adapt(source,profile,output):
    source=Path(source).resolve();info=json.loads((source/'meta/info.json').read_text());c=json.loads(Path(profile).read_text())
    if 'split' not in c:
        splits=info.get('splits',{})
        if len(splits)!=1:raise ValueError('Source has multiple/unknown splits; adapter must select one original split explicitly')
        c['split']=next(iter(splits))
    source_splits=c.get('source_split_ranges',info.get('splits',{}))
    if c['split'] not in source_splits:raise ValueError('Requested split is absent from source splits')
    split_range=source_splits[c['split']]
    if not isinstance(split_range,str) or ':' not in split_range:raise ValueError('Source split must be an episode start:end range; adapt other layouts explicitly')
    split_start,split_end=split_range.split(':');split_start=int(split_start or 0);split_end=int(split_end or info['total_episodes'])
    version=info['codebase_version'];refs=defaultdict(list);frameids=defaultdict(list);episode_task_ids=defaultdict(set)
    for p in sorted((source/'data').rglob('*.parquet')):
        t=pq.ParquetFile(p).read(columns=['episode_index','frame_index','task_index']);task_ids=t['task_index'].to_numpy();ep=t['episode_index'].to_numpy();frames=t['frame_index'].to_numpy();cuts=np.r_[0,np.flatnonzero(ep[1:]!=ep[:-1])+1,len(ep)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            key=int(ep[a]);refs[key].append({'path':str(p),'offset':int(a),'length':int(b-a)});frameids[key].append(frames[a:b]);episode_task_ids[key].update(map(int,np.unique(task_ids[a:b])))
    if version=='v2.1':rows=[json.loads(x) for x in (source/'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
    elif version=='v3.0':
        rows=[]
        for p in sorted((source/'meta/episodes').rglob('*.parquet')):
            pf=pq.ParquetFile(p);names=[k for k in pf.schema_arrow.names if k in ['episode_index','length'] or k.startswith('videos/')];rows.extend(pf.read(columns=names).to_pylist())
    else:raise ValueError('Only v2.1/v3 source layouts supported')
    if (source/'meta/tasks.parquet').exists():v=pq.ParquetFile(source/'meta/tasks.parquet').read().to_pylist()
    else:v=[json.loads(x) for x in (source/'meta/tasks.jsonl').read_text().splitlines() if x.strip()]
    names=next((k for k in ['task','__index_level_0__'] if v and k in v[0]),'task');words={r['task_index']:r[names] for r in v}
    cameras=[k for k,f in info['features'].items() if f.get('dtype')=='video'];episodes=[]
    for row in sorted(rows,key=lambda r:r['episode_index']):
        if not split_start<=int(row['episode_index'])<split_end:continue
        ep=int(row['episode_index']);n=int(row['length']);ids=np.concatenate(frameids[ep])
        if not np.array_equal(ids,np.arange(n)):raise ValueError('Source episode has row gaps or conflicting frame indices')
        videos={}
        for key in cameras:
            cam=key.removeprefix('observation.images.')
            if version=='v3.0':
                prefix='videos/'+key+'/';p=source/info['video_path'].format(video_key=key,chunk_index=row[prefix+'chunk_index'],file_index=row[prefix+'file_index'])
                videos[cam]={'path':str(p),'from_timestamp':row[prefix+'from_timestamp'],'to_timestamp':row[prefix+'to_timestamp'],'length':n,'fps':info['fps']}
            else:
                p=source/info['video_path'].format(video_key=key,episode_index=ep,episode_chunk=ep//int(info.get('chunks_size',1000)))
                videos[cam]={'path':str(p),'length':n,'fps':info['fps']}
        if not episode_task_ids[ep].issubset(words):raise ValueError('Source vocabulary missing task id; prove a repair from original instructions first')
        if not all(isinstance(words[i],str) for i in episode_task_ids[ep]):raise ValueError('Source vocabulary not decoded text')
        episodes.append({'source_episode_id':ep,'data':refs[ep],'videos':videos,'length':n,'fps':info['fps'],'split_group_uid':f"{c['source']['id']}:source-episode:{ep}"})
    c['task_id_column']='task_index';c['task_vocabulary']={str(k):v for k,v in words.items()}
    c['episodes']=episodes;c['source'].update(root=str(source),source_codebase_version=version)
    if c['media']['policy']=='native':c['fps']=info['fps']
    Path(output).write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');return {'output':str(output),'episodes':len(episodes)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('profile');p.add_argument('output');a=p.parse_args();print(json.dumps(adapt(a.source,a.profile,a.output)))
