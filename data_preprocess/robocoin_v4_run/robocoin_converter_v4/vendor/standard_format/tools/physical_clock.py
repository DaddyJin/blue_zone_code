"""Optional original acquisition clock, aligned explicitly to retained source rows."""
import hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def load_clock(config,episodes):
 spec=config.get('physical_clock')
 if spec is None:return None
 p=Path(spec['path'])
 if not spec.get('evidence') or spec.get('kind')!='original_acquisition_clock':raise ValueError('Physical clock provenance is required')
 if hashlib.sha256(p.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Physical clock sidecar changed')
 t=pq.ParquetFile(p).read(columns=['episode_index','frame_index','timestamp_ns'])
 for k in t.column_names:
  if t[k].null_count or t[k].type!=pa.int64():raise ValueError('Clock columns must be non-null int64: '+k)
 ep=t['episode_index'].to_numpy();fi=t['frame_index'].to_numpy();ns=t['timestamp_ns'].to_numpy();result={}
 for e in episodes:
  key=int(e['source_episode_id']);m=np.flatnonzero(ep==key);m=m[np.argsort(fi[m],kind='stable')]
  if not np.array_equal(fi[m],np.arange(int(e['length']))):raise ValueError('Physical clock must cover every retained row exactly once: '+str(key))
  a=ns[m]
  if len(a)>1 and np.any(a[1:]<=a[:-1]):raise ValueError('Acquisition clock must strictly increase within each episode')
  result[key]=a
 return result
