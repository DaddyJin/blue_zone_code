"""Float64 exact-PTS reference reader, including multiple cameras and video-only data."""
from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import json,math,os
import av
import numpy as np
import pyarrow.parquet as pq

def rgb_at_pts(container,target):
    s=container.streams.video[0]
    if not s.codec_context.is_open:s.codec_context.thread_count=1
    tb=float(s.time_base);target=float(target)
    if not math.isfinite(target):raise ValueError('Nonfinite target PTS')
    # Reconstruct exact integer tick; floor(float/tb) can land one tick early.
    container.seek(round(target/tb),stream=s,backward=True,any_frame=False)
    tolerance=max(1e-8,tb*1e-5)
    for count,frame in enumerate(container.decode(s)):
        if count>10000:raise RuntimeError('Seek decode guard exceeded')
        if frame.pts is None:raise ValueError('Frame has no PTS')
        actual=float(frame.pts*frame.time_base)
        if abs(actual-target)<=tolerance:return frame.to_ndarray(format='rgb24')
        if actual>target+tolerance:raise ValueError(f'Exact target PTS absent: {target} -> {actual}')
    raise EOFError('Target frame not decoded')

class StandardDataset:
    """Map-style NumPy dataset; caches/handles reset in each DataLoader worker."""
    def __init__(self,root,episode_indices=None,include_invalid=False):
        self.root=Path(root);self.info=json.loads((self.root/'meta/info.json').read_text());self.pid=os.getpid()
        self.side={}
        with (self.root/'meta/episodes.jsonl').open() as f:
            for line in f:
                e=json.loads(line);self.side[e['episode_index']]=e
        self.rows=[]
        for p in sorted((self.root/'meta/episodes').rglob('*.parquet')):self.rows.extend(pq.ParquetFile(p).read().to_pylist())
        allowed=None if episode_indices is None else set(map(int,episode_indices))
        self.rows=[r for r in sorted(self.rows,key=lambda r:r['episode_index']) if (allowed is None or r['episode_index'] in allowed) and (include_invalid or self.side[r['episode_index']]['video_valid'])]
        self.ends=np.cumsum([r['length'] for r in self.rows]);self.tables=OrderedDict();self.videos=OrderedDict();self.axes=OrderedDict()
        self.timing={(r['episode_index'],r['camera']):r for r in pq.ParquetFile(self.root/'meta/video_timing/episodes.parquet').read().to_pylist()}
        self.tasks=self._words('tasks','task');self.subtasks=self._words('subtasks','subtask')
    def _words(self,file,label):return {r[label+'_index']:r[label] for r in pq.ParquetFile(self.root/f'meta/{file}.parquet').read().to_pylist()}
    def __len__(self):return int(self.ends[-1]) if len(self.ends) else 0
    def __getitem__(self,index):
        if self.pid!=os.getpid():self.close();self.pid=os.getpid()
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        position=int(np.searchsorted(self.ends,index,side='right'));e=self.rows[position];local=index-(int(self.ends[position-1]) if position else 0)
        p=self.root/self.info['data_path'].format(chunk_index=e['data/chunk_index'],file_index=e['data/file_index'])
        if p not in self.tables:
            self.tables[p]=pq.ParquetFile(p).read()
            if len(self.tables)>2:self.tables.popitem(last=False)
        self.tables.move_to_end(p);row=self.tables[p].slice(e['data/file_from_offset']+local,1).to_pylist()[0]
        if row['episode_index']!=e['episode_index'] or row['frame_index']!=local:raise ValueError('Row ownership mismatch')
        result={k:np.asarray(v,np.float32) if isinstance(v,list) and k in ('state_unified','action_unified','mask_state','mask_action','reliability') else v for k,v in row.items()}
        for key,f in self.info['features'].items():
            if f['dtype']!='video':continue
            cam=key.removeprefix('observation.images.');timing=self.timing[(e['episode_index'],cam)]
            if not timing['video_valid']:raise ValueError('Video-invalid episode is not readable')
            prefix='videos/'+key+'/'
            v=self.root/self.info['video_path'].format(video_key=key,chunk_index=e[prefix+'chunk_index'],file_index=e[prefix+'file_index'])
            axispath=self.root/timing['pts_path']
            if axispath not in self.axes:
                self.axes[axispath]=np.load(axispath,mmap_mode='r',allow_pickle=False)
                if len(self.axes)>4:self.axes.popitem(last=False)
            target=self.axes[axispath][timing['video_frame_offset']+local]
            if v not in self.videos:
                self.videos[v]=av.open(str(v))
                if len(self.videos)>2:self.videos.popitem(last=False)[1].close()
            self.videos.move_to_end(v);result[key]=rgb_at_pts(self.videos[v],target);result[key+'.pts']=float(target)
        result['task']=self.tasks[row['task_index']];result['subtask']=self.subtasks.get(row['subtask_index'],'')
        result['robot_type']=self.info['robot_type'];result['profile_id']=self.info['custom']['profile_id'];result['split_group_uid']=self.side[e['episode_index']]['split_group_uid']
        return result
    def close(self):
        for c in self.videos.values():c.close()
        self.videos.clear();self.tables.clear();self.axes.clear()
    def __getstate__(self):
        state=self.__dict__.copy();state.update(videos=OrderedDict(),tables=OrderedDict(),axes=OrderedDict());return state
