"""Dataset-specific adapter -> episode manifest -> Unified55 LeRobot v3."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse,hashlib,json,shutil,uuid
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from common import VERSION,SLOTS,VECTORS,Stats,dump,write_table,values,fixed55,map55,time_columns,feature
import media

class SourceCache:
    def __init__(self):self.cache=OrderedDict()
    def get(self,path):
        if path not in self.cache:
            self.cache[path]=pq.ParquetFile(path).read()
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(path);return self.cache[path]
    def episode(self,ref,base,n):
        if ref is None:return pa.table({'_frame_placeholder':pa.array(np.arange(n,dtype=np.int64))})
        refs=ref if isinstance(ref,list) else [ref];tables=[]
        for r in refs:
            spec={'path':r} if isinstance(r,str) else r;t=self.get((base/spec['path']).resolve())
            if 'episode_index' in spec:
                import pyarrow.compute as pc
                t=t.filter(pc.equal(t['episode_index'],int(spec['episode_index'])))
            else:t=t.slice(int(spec.get('offset',0)),int(spec.get('length',len(t))))
            tables.append(t)
        result=pa.concat_tables(tables)
        if len(result)!=n:raise ValueError(f'{len(result)} source rows vs {n} video frames; adapter must align streams explicitly')
        return result

class Vocabulary:
    def __init__(self):self.words=[];self.ids={}
    def id(self,text):
        if not isinstance(text,str):raise ValueError('Adapter must decode language to str, not byte arrays or str(array)')
        if text not in self.ids:self.ids[text]=len(self.words);self.words.append(text)
        return self.ids[text]
    def write(self,path,label):
        frame=pd.DataFrame({label+'_index':np.arange(len(self.words),dtype=np.int64)},index=pd.Index(self.words,name=label,dtype='str'))
        write_table(path,pa.Table.from_pandas(frame,preserve_index=True))

def calibrate(table,config,mappings,columns):
    for cam,fields in config.get('calibration_columns',{}).items():
        H=np.asarray(mappings[cam]['image_transform'],np.float64)
        for kind,source in fields.items():
            if kind not in ('K','P'):raise ValueError('Only explicit K(3x3)/P(3x4) image-coordinate updates are supported')
            a=values(table[source]);width=3 if kind=='K' else 4
            a=np.asarray(a,np.float64).reshape(len(table),3,width)
            transformed=np.einsum('ij,njk->nik',H,a).reshape(len(table),-1).astype(np.float32)
            columns[f'observation.calibration.{cam}.{kind}']=pa.FixedSizeListArray.from_arrays(pa.array(transformed.ravel()),3*width)

def convert(manifest, output):
    manifest=Path(manifest).resolve();base=manifest.parent;c=json.loads(manifest.read_text());out=Path(output).resolve()
    from check_manifest import check
    check(c)
    c.setdefault('source',{}).setdefault('manifest_base',str(base))
    for k in ('dataset_id','profile_id','data_kind','robot_type','fps','media','action','state_fields','source','episodes'):
        if k not in c:raise ValueError('Missing manifest '+k)
    if not all(isinstance(c['source'].get(k),str) and c['source'][k] for k in ('id','revision')):raise ValueError('source.id and source.revision required; mark genuinely unknown revisions explicitly')
    if c['data_kind'] not in ('robot','ego','observation_only','video_only'):raise ValueError('Invalid data_kind')
    mode=c['action']['mode']
    if mode not in ('next_state','source','none'):raise ValueError('Choose action.mode: next_state/source/none')
    if mode=='source' and c['action'].get('terminal') not in ('keep','mask'):raise ValueError('Source actions require explicit terminal keep/mask')
    if mode=='source' and not c['action'].get('fields'):raise ValueError('Source action mappings are empty')
    if not c['episodes'] or not np.isfinite(float(c['fps'])) or float(c['fps'])<=0:raise ValueError('Empty episodes or invalid fps')
    if c['data_kind']=='video_only' and (c['state_fields'] or mode!='none' or any(not e.get('videos') for e in c['episodes'])):raise ValueError('video_only requires video, empty state_fields and action.mode=none')
    for f in c['state_fields']+c['action'].get('fields',[]):
        if f.get('owner')!=c['robot_type']:raise ValueError('Every 55D field must explicitly belong to this unit robot_type; retain other-robot fields as passthrough')
        if f['slot'] in ('left_eef','right_eef','head','mobile') and f.get('frame') not in c.get('frames',{}):raise ValueError('Declare the field frame in frames, including unknown frames')
    if out.exists():raise FileExistsError('Output already exists; never modifies a prior dataset')
    stage=out.with_name('.'+out.name+'.building-'+uuid.uuid4().hex);stage.mkdir(parents=True)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    try:
        from physical_clock import load_clock
        clocks=load_clock(c,c['episodes'])
        m=media.prepare(c,c['episodes'],base,stage);pts,decode_report=media.decode_timing(stage,m)
        vocab=Vocabulary();subvocab=Vocabulary();cache=SourceCache();stats=Stats();epmeta=[];side=[];timing=[];buffer=[];buffer_rows=0;file_number=0;global_row=0
        features=None;maximum_rows=int(c.get('parquet_rows',50000))
        def flush():
            nonlocal buffer,buffer_rows,file_number
            if not buffer:return
            table=pa.concat_tables(buffer);chunk,fi=divmod(file_number,1000)
            write_table(stage/f'data/chunk-{chunk:03d}/file-{fi:03d}.parquet',table)
            buffer=[];buffer_rows=0;file_number+=1
        for ep,e in enumerate(c['episodes']):
            n=m['lengths'][ep];t=cache.episode(e.get('data'),base,n);maps=m['episodes'][ep]
            if clocks is not None:t=t.append_column('_physical_timestamp_ns',pa.array(clocks[int(e['source_episode_id'])],pa.int64()))
            if buffer and buffer_rows+n>maximum_rows:flush()
            state,ms=map55(t,c['state_fields']);valid=np.ones(n,bool)
            if e.get('excluded'):valid[:]=False
            if c.get('invalid_sample_column'):valid&=~values(t[c['invalid_sample_column']]).astype(bool)
            ms[~valid]=0
            if mode=='next_state':
                action=np.concatenate([state[1:],state[-1:]]);ma=np.concatenate([ms[1:],np.zeros((1,55),np.float32)])
            elif mode=='source':
                action,ma=map55(t,c['action']['fields'])
                if c['action']['terminal']=='mask':ma[-1]=0
            else:action=np.zeros_like(state);ma=np.zeros_like(ms)
            action_semantics={'mode':mode,'reason':'configured'}
            if mode=='source' and c['action'].get('episode_copy_policy'):
                from episode_policy import apply_episode_policy
                action,ma,action_semantics=apply_episode_policy(t,c,state,ms,action,ma)
            effective_next=mode=='next_state' or action_semantics['mode']=='derived_next_state'
            ma[~valid]=0
            times,clock=time_columns(t,e.get('time',c.get('time',{'kind':'unknown'})),n)
            dt=times['source_dt'].copy() if effective_next else np.zeros(n,np.float64)
            dtmask=times['mask_source_dt'].copy() if effective_next else np.zeros(n,np.float32)
            if mode=='source' and c['action'].get('dt_column'):
                if c['action'].get('dt_kind')!='measured':raise ValueError('action_dt must be measured; retain nominal values separately')
                dt=np.asarray(values(t[c['action']['dt_column']]),np.float64);dtmask=(np.isfinite(dt)&(dt>0)).astype(np.float32);dt=np.where(dtmask,dt,0)
            dtmask*=ma.any(1);dt=np.where(dtmask,dt,0)
            if c.get('task_column'):tasktexts=t[c['task_column']].to_pylist()
            elif c.get('task_id_column'):tasktexts=[c['task_vocabulary'][str(int(i))] for i in values(t[c['task_id_column']])]
            else:tasktexts=[e.get('task','')]*n
            taskids=np.array([vocab.id(x) for x in tasktexts],np.int64);subids=np.full(n,-1,np.int64)
            segments=e.get('segments',[]);previous_end=0
            for seg in segments:
                a,b=int(seg['start_frame']),int(seg['end_frame'])
                if not (0<=a<b<=n) or a<previous_end:raise ValueError('Subtask segments must be sorted, nonoverlapping [start,end)')
                if 'skill' not in seg:seg['skill']=None
                subids[a:b]=subvocab.id(seg['action_text']);previous_end=b
            if maps:
                main=next(iter(maps));mainmap=maps[main];axis=pts[mainmap['file']][0];a=mainmap['video_frame_offset']
                timestamp=axis[a:a+n]-axis[a]
            else:timestamp=times['source_timestamp'].copy() if clock['kind'] in ('measured','published') else np.arange(n,dtype=np.float64)/float(c['fps'])
            columns={};passthrough=c.get('passthrough_columns',[])
            if passthrough=='all':passthrough=t.column_names if e.get('data') is not None else []
            for name in passthrough:
                columns['source.'+name]=t[name]
            columns.update(index=pa.array(np.arange(global_row,global_row+n,dtype=np.int64)),episode_index=pa.array(np.full(n,ep,np.int64)),frame_index=pa.array(np.arange(n,dtype=np.int64)),timestamp=pa.array(timestamp,pa.float64()),task_index=pa.array(taskids),subtask_index=pa.array(subids),mask_language=pa.array([bool(x.strip()) for x in tasktexts]),qc_sample_valid=pa.array(valid),action_dt=pa.array(dt),mask_action_dt=pa.array(dtmask))
            for name,x in times.items():columns[name]=pa.array(x)
            base_time=times['source_timestamp'] if clock['kind']=='measured' else np.asarray(values(t['timestamp']),np.float64)
            sample_dt=np.r_[np.diff(base_time),0.];sample_good=np.r_[np.isfinite(np.diff(base_time))&(np.diff(base_time)>0),False]
            if not sample_good[:-1].all():raise ValueError('Published/physical clock has invalid intervals')
            columns['sample_dt']=pa.array(sample_dt,pa.float64());columns['mask_sample_dt']=pa.array(sample_good.astype(np.float32));columns['sample_dt_kind']=pa.array(np.where(sample_good,2 if clock['kind']=='measured' else 1,0),pa.int8())
            if c['action'].get('episode_copy_policy'):
                # 0=publisher action retained, 1=measured next-state derived. Terminal code remains 1, mask=0.
                columns['action_source_kind']=pa.array(np.full(n,int(effective_next),np.int8))
                columns['action_target_step']=pa.array(np.where(np.arange(n)<n-1,1,0) if effective_next else np.zeros(n),pa.int8())
            vectors={'state_unified':state,'action_unified':action,'mask_state':ms,'mask_action':ma,'reliability':ms.copy()}
            for name,x in vectors.items():
                columns[name]=fixed55(x);stats.add(name,x,ms if name=='state_unified' else ma if name=='action_unified' else None)
            calibrate(t,c,maps,columns)
            table=pa.table(columns)
            current_features={f.name:feature(table[f.name]) for f in table.schema}
            if features is None:features=current_features
            if features!=current_features:raise ValueError('Column schemas differ between episodes; adapter must supply consistent schema')
            for key in ('timestamp','source_timestamp','source_dt','mask_source_dt','action_dt','mask_action_dt','sample_dt','mask_sample_dt','sample_dt_kind','index','episode_index','frame_index','task_index','subtask_index'):stats.add(key,values(table[key]))
            row={'episode_index':ep,'tasks':list(dict.fromkeys(tasktexts)),'length':n,'dataset_from_index':global_row,'dataset_to_index':global_row+n,'data/chunk_index':file_number//1000,'data/file_index':file_number%1000,'data/file_from_offset':buffer_rows,'data/file_to_offset':buffer_rows+n,'meta/episodes/chunk_index':0,'meta/episodes/file_index':0}
            for cam,v in maps.items():
                prefix=f'videos/observation.images.{cam}/'
                for key in ('chunk_index','file_index','from_timestamp','to_timestamp'):row[prefix+key]=v[key]
                timing.append({'episode_index':ep,'camera':cam,'video_frame_offset':v['video_frame_offset'],'length':n,'pts_path':v['pts_path'],'video_valid':not e.get('excluded',False)})
            epmeta.append(row)
            side.append({'episode_index':ep,'source_episode_id':e.get('source_episode_id',ep),'length':n,'tasks':row['tasks'],'action_config':segments,'action_semantics':action_semantics,'annotation_granularity':'subtask' if segments else 'episode_task_only' if any(tasktexts) else 'unlabeled','profile_id':c['profile_id'],'split':e.get('split',c.get('split','train')),'split_group_uid':e.get('split_group_uid',f"{c['source'].get('id',c['dataset_id'])}:episode:{e.get('source_episode_id',ep)}"),'source_time':clock,'videos':maps,'video_valid':not e.get('excluded',False),'source':e.get('source',{}),'source_data':e.get('data'),'source_columns':[{'name':f.name,'arrow_type':str(f.type)} for f in t.schema] if e.get('data') is not None else []})
            buffer.append(table);buffer_rows+=n;global_row+=n
        flush();write_table(stage/'meta/episodes/chunk-000/file-000.parquet',pa.Table.from_pylist(epmeta));write_table(stage/'meta/video_timing/episodes.parquet',pa.Table.from_pylist(timing))
        vocab.write(stage/'meta/tasks.parquet','task');subvocab.write(stage/'meta/subtasks.parquet','subtask')
        with (stage/'meta/episodes.jsonl').open('w') as f:
            for x in side:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
        for cam in c['episodes'][0]['videos']:
            h=next(f['header'] for f in m['files'] if f['cam']==cam)
            features['observation.images.'+cam]={'dtype':'video','shape':[h['height'],h['width'],3],'names':['height','width','channels'],'info':{'video.height':h['height'],'video.width':h['width'],'video.codec':h['codec_name'],'video.pix_fmt':h['pix_fmt'],'video.fps':float(c['fps']),'video.channels':3,'video.is_depth_map':False,'has_audio':False}}
        split_names=set(x['split'] for x in side)
        if len(split_names)!=1:raise ValueError('One source split per delivery unit; split manifests externally, keep group ids')
        info={'codebase_version':'v3.0','robot_type':c['robot_type'],'fps':float(c['fps']),'total_frames':global_row,'total_episodes':len(epmeta),'total_tasks':len(vocab.words),'total_videos':len(m['files']),'total_chunks':max(1,(file_number+999)//1000),'chunks_size':1000,'data_files_size_in_mb':100,'video_files_size_in_mb':float(c['media'].get('max_file_bytes',500_000_000))/1e6,'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet','video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4','splits':{next(iter(split_names)):f'0:{len(side)}'},'features':features,'custom':{'standard_format_version':VERSION,'dataset_id':c['dataset_id'],'data_kind':c['data_kind'],'profile_id':c['profile_id'],'source':c['source'],'state_fields':c['state_fields'],'action':c['action'],'frames':c.get('frames',{}),'media':c['media'],'semantic_notes':c.get('semantic_notes',{}),'physical_clock':c.get('physical_clock'),'time_axis':{'sample_dt':'Adjacent retained-row interval; sample_dt_kind=1 published grid (not measured), 2 original physical clock, 0 terminal/invalid. mask_sample_dt only indicates valid numeric interval.','timestamp':'video query time relative to each episode main-camera start, or sample time if no video; per-camera actual PTS is authoritative','source_timestamp':'source clock relative to sidecar raw_origin; kind is per episode','source_dt':'physical sample interval only when kind=measured and mask_source_dt=1; otherwise 0','action_dt':'measured action horizon, otherwise 0 with mask 0'},'reliability':'binary copy of mask_state, not calibrated confidence','statistics':{'55d':'masked component-wise','images':'not computed, never fabricated; compute image normalization separately if required'},'calibration_columns':c.get('calibration_columns',{}),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'passthrough_columns':c.get('passthrough_columns',[])}}
        dump(stage/'meta/info.json',info);dump(stage/'meta/stats.json',stats.export());dump(stage/'meta/input_manifest.json',c);dump(stage/'meta/media_manifest.json',m)
        dump(stage/'meta/qc/decode.json',{'passed':True,'files':decode_report})
        if (stage/'.work').exists():shutil.rmtree(stage/'.work')
        from validate import validate
        report=validate(stage,decode=False,samples=3)
        dump(stage/'qc_report.json',report)
        if not report['passed']:raise ValueError('Output validation failed: '+str(report['errors'][:5]))
        dump(stage/'SUCCESS.json',{'status':'passed','standard_format_version':VERSION,'frames':global_row,'episodes':len(side),'manifest_sha256':info['custom']['manifest_sha256'],'qc':'qc_report.json'})
        stage.rename(out)
        return {'output':str(out),'frames':global_row,'episodes':len(side),'video_files':len(m['files']),'passed':True}
    except Exception:
        # Keep failed staging for diagnosis; no success marker, no source mutation.
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('output');a=p.parse_args();print(json.dumps(convert(a.manifest,a.output),ensure_ascii=False))
