"""55D mapping and physical-time helpers. No dataset-specific inference."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = '2.1-robocoin'
SLOTS = {'left_arm_joint':(0,7), 'right_arm_joint':(7,14),
 'left_eef':(14,21), 'right_eef':(21,28), 'left_gripper':(28,29),
 'right_gripper':(29,30), 'left_hand':(30,36), 'right_hand':(36,42),
 'waist':(42,46), 'head':(46,48), 'mobile':(48,51), 'reserved':(51,55)}
VECTORS = ('state_unified','action_unified','mask_state','mask_action','reliability')

def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def write_table(path, table):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');pq.write_table(table,tmp,compression='zstd',compression_level=1);tmp.replace(path)

def values(column):
    a=column.combine_chunks() if isinstance(column,pa.ChunkedArray) else column
    if pa.types.is_fixed_size_list(a.type):
        return np.asarray(a.values.to_numpy(zero_copy_only=False))[a.offset*a.type.list_size:(a.offset+len(a))*a.type.list_size].reshape(len(a),a.type.list_size)
    if pa.types.is_list(a.type) or pa.types.is_large_list(a.type):
        offsets=np.asarray(a.offsets);sizes=np.diff(offsets)
        if len(sizes) and np.any(sizes!=sizes[0]):raise ValueError('Ragged values require an explicit adapter')
        width=int(sizes[0]) if len(sizes) else 0
        return np.asarray(a.values.slice(int(offsets[0]),int(offsets[-1]-offsets[0])).to_numpy(zero_copy_only=False)).reshape(len(a),width)
    return np.asarray(a.to_numpy(zero_copy_only=False))

def fixed55(a):
    return pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(a,np.float32).ravel()),55)

def gripper(raw, rule):
    """Output dimensionless opening. No assumed width calibration or clipping."""
    x=np.asarray(raw,np.float64);mode=rule['mode'];valid=np.isfinite(x)
    if mode=='native_open_direction':
        scale=float(rule['scale']);offset=float(rule.get('offset',0))
        if scale not in (-1.,1.) or not rule.get('direction_evidence'):raise ValueError('Native gripper requires sign +/-1 and direction evidence')
        y=scale*x+offset
        return np.where(valid,y,0),valid
    if mode=='open_01':y=x
    elif mode=='closed_01':y=1-x
    elif mode=='width':
        closed=float(rule['closed']);opened=float(rule['open'])
        if opened<=closed:raise ValueError('Gripper open endpoint must exceed closed endpoint')
        if rule.get('unit') not in ('m','mm'):raise ValueError('Width requires explicit m/mm unit and endpoints in that unit')
        y=(x-closed)/(opened-closed)
    elif mode=='unknown':return np.zeros_like(x),np.zeros_like(valid)
    else:raise ValueError('Unsupported gripper rule: '+mode)
    valid&=(y>=0)&(y<=1)
    return np.where(valid,y,0),valid

def map55(table, fields):
    n=len(table);out=np.zeros((n,55),np.float32);mask=np.zeros_like(out);used=set()
    for f in fields:
        slot=f['slot']
        if slot not in SLOTS or slot=='reserved':raise ValueError('Invalid/reserved mapping slot: '+slot)
        for key in ('source','quantity','unit','provenance'):
            if not f.get(key):raise ValueError(f'{slot}: missing mapping {key}')
        if f['provenance'] not in ('measured','commanded','derived','assumed','unknown'):raise ValueError('Bad provenance')
        if f['provenance']=='derived' and not f.get('method'):raise ValueError('Derived field requires method')
        a=np.array(values(table[f['source']]),dtype=np.float64,copy=True)
        if a.ndim==1:a=a[:,None]
        if 'indices' in f:a=a[:,f['indices']]
        start,end=SLOTS[slot];components=f.get('components',list(range(a.shape[1])))
        if len(components)!=a.shape[1] or len(set(components))!=len(components) or any(c<0 or c>=end-start for c in components):raise ValueError('Mapping components/width invalid: '+slot)
        targets=[start+c for c in components]
        if used.intersection(targets):raise ValueError('Overlapping 55D mappings')
        used.update(targets);valid=np.isfinite(a)
        if slot.endswith('_eef'):
            if not f.get('frame') or not f.get('reference_point'):raise ValueError('EEF requires frame id and tracked/TCP reference_point')
            if f['unit']!='m' or f.get('source_unit',f['unit']) not in ('m','mm'):raise ValueError('EEF canonical unit=m; source_unit=m/mm')
            for j,c in enumerate(components):
                if c<3 and f.get('source_unit',f['unit'])=='mm':a[:,j]*=.001
            qcols=[components.index(c) for c in range(3,7) if c in components]
            if qcols:
                if len(qcols)!=4 or f.get('quaternion_order') not in ('xyzw','wxyz'):raise ValueError('Quaternion requires all four components and explicit xyzw/wxyz')
                q=a[:,qcols].copy()
                if f['quaternion_order']=='wxyz':q=q[:,[1,2,3,0]]
                if f.get('quaternion_recovery'):
                    from quaternion_recovery import recover_quaternion
                    q,good,_=recover_quaternion(q)
                else:
                    norm=np.linalg.norm(q,axis=1);good=np.isfinite(q).all(1)&(norm>1e-8)
                    if 'quaternion_norm_tolerance' in f:
                        tol=float(f['quaternion_norm_tolerance'])
                        if not 0<tol<1:raise ValueError('Quaternion norm tolerance must be between 0 and 1')
                        good &= np.abs(norm-1)<=tol
                    q[good]/=norm[good,None];q[~good]=0
                    for i in range(1,n):
                        if good[i] and good[i-1] and np.dot(q[i],q[i-1])<0:q[i]*=-1
                a[:,qcols]=q;valid[:,qcols]=good[:,None]
        elif slot.endswith('_gripper'):
            if a.shape[1]!=1:raise ValueError('Gripper must be scalar')
            if f['gripper']['mode']!='native_open_direction' and f['unit']!='1':raise ValueError('Normalized gripper must have unit 1')
            a[:,0],valid[:,0]=gripper(a[:,0],f['gripper'])
        elif slot.endswith('_joint') or slot.endswith('_hand') or slot=='head':
            source_unit=f.get('source_unit',f['unit'])
            if (source_unit,f['unit']) in [('deg','rad'),('deg/s','rad/s')]:a*=np.pi/180
            elif source_unit!=f['unit'] or f['unit'] not in ('rad','rad/s'):raise ValueError('Joint/head output unit must be rad/rad/s; explicit source_unit may be deg/deg/s')
            if len(f.get('component_names',[]))!=len(components):raise ValueError('Joint/head mappings require ordered component_names')
        else:
            if len(f.get('component_names',[]))!=len(components):raise ValueError('Waist/mobile require ordered component_names')
            units=f['unit'] if isinstance(f['unit'],list) else [f['unit']]*len(components)
            if len(units)!=len(components) or any(u not in ('m','rad','m/s','rad/s') for u in units):raise ValueError('Waist/mobile require explicit SI units; convert in adapter')
        if f.get('valid_column'):
            v=values(table[f['valid_column']]);valid&=np.asarray(v,bool).reshape(n,-1)
        if f['provenance'] in ('unknown','assumed'):valid[:]=False
        out[:,targets]=np.where(valid,a,0).astype(np.float32);mask[:,targets]=valid
    return out,mask

def time_columns(table, spec, n):
    kind=spec.get('kind','unknown')
    if kind not in ('measured','published','nominal','video_pts','unknown'):raise ValueError('Unknown time provenance')
    if spec.get('column'):
        raw=values(table[spec['column']]);unit=spec.get('unit','s')
        if unit not in ('s','ms','us','ns'):raise ValueError('Unsupported clock unit')
        factor={'s':1.,'ms':1e-3,'us':1e-6,'ns':1e-9}[unit]
        # Difference integer clocks before float conversion to avoid epoch-ns precision loss.
        if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.number):raise ValueError('Time column must be a numeric scalar')
        if np.issubdtype(raw.dtype,np.unsignedinteger):
            if np.any(raw>np.iinfo(np.int64).max):raise ValueError('Unsigned clock exceeds int64; adapter must subtract origin first')
            raw=raw.astype(np.int64)
        finite_raw=np.isfinite(raw);first=int(np.flatnonzero(finite_raw)[0]) if finite_raw.any() else 0
        origin=raw[first].item() if n and finite_raw.any() else 0
        t=np.asarray(raw-origin,np.float64)*factor
    elif kind=='nominal':
        fps=float(spec['fps'])
        if not np.isfinite(fps) or fps<=0:raise ValueError('Nominal source fps must be positive')
        t=np.arange(n,dtype=np.float64)/fps;origin=0
    elif kind=='unknown':t=np.zeros(n,np.float64);origin=None
    else:raise ValueError('Non-unknown source time requires its real column')
    finite=np.isfinite(t);t=np.where(finite,t,0)
    dt=np.zeros(n,np.float64);good=np.zeros(n,bool)
    if n>1 and kind=='measured':
        diff=np.diff(t);good[:-1]=finite[:-1]&finite[1:]&(diff>0);dt[:-1]=np.where(good[:-1],diff,0)
    return {'source_timestamp':t,'mask_source_timestamp':(finite & (kind!='unknown')).astype(np.float32),'source_dt':dt,'mask_source_dt':good.astype(np.float32)}, {'kind':kind,'raw_origin':origin,'raw_unit':spec.get('unit','s'),'origin_rule':'source_timestamp is relative to this origin; raw source column retained when selected'}

class Stats:
    def __init__(self):self.data={}
    def add(self,key,a,mask=None):
        a=np.asarray(a,np.float64)
        if a.ndim==1:a=a[:,None]
        good=np.isfinite(a) if mask is None else np.asarray(mask,bool)&np.isfinite(a)
        state=self.data.setdefault(key,[np.zeros(a.shape[1]),np.zeros(a.shape[1]),np.zeros(a.shape[1]),np.full(a.shape[1],np.inf),np.full(a.shape[1],-np.inf)])
        state[0]+=good.sum(0);state[1]+=np.where(good,a,0).sum(0);state[2]+=np.where(good,a*a,0).sum(0)
        state[3]=np.minimum(state[3],np.where(good,a,np.inf).min(0));state[4]=np.maximum(state[4],np.where(good,a,-np.inf).max(0))
    def export(self):
        result={}
        for key,(count,total,sq,lo,hi) in self.data.items():
            mean=total/np.maximum(count,1);var=np.maximum(0,sq/np.maximum(count,1)-mean*mean)
            result[key]={'min':np.where(count,lo,0).tolist(),'max':np.where(count,hi,0).tolist(),'mean':mean.tolist(),'std':np.sqrt(var).tolist(),'count':count.astype(int).tolist()}
        return result

def feature(column):
    t=column.type;shape=[1]
    if pa.types.is_fixed_size_list(t):shape=[t.list_size];t=t.value_type
    elif pa.types.is_list(t):
        a=column.combine_chunks();width=np.diff(np.asarray(a.offsets))
        if len(width) and not np.all(width==width[0]):raise ValueError('Ragged passthrough must be adapted or excluded explicitly')
        shape=[int(width[0]) if len(width) else 0];t=t.value_type
    types={pa.float32():'float32',pa.float64():'float64',pa.int64():'int64',pa.int32():'int32',pa.uint8():'uint8',pa.int8():'int8',pa.int16():'int16',pa.uint16():'uint16',pa.uint32():'uint32',pa.uint64():'uint64',pa.bool_():'bool',pa.string():'string'}
    if t not in types:raise ValueError('Unsupported passthrough dtype: '+str(t))
    return {'dtype':types[t],'shape':shape,'names':None}
