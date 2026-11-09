"""Streaming per-episode source equality, independent mappings, packet hashes and RGB reads."""
from pathlib import Path
import argparse,json,sys,traceback,hashlib,re
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,av
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'vendor/standard_format/tools'))
from convert_repos import digest
from convert import SourceCache
from read_frame import StandardDataset

def arr(t,k):return np.asarray(t[k].to_pylist())
def packets(paths):
 h=hashlib.sha256();count=0
 for path in paths:
  with av.open(str(path)) as c:
   for p in c.demux(c.streams.video[0]):
    if p.size:h.update(p.size.to_bytes(8,'big'));h.update(bytes(p));count+=1
 return count,h.hexdigest()
def check_mapping(src,dst,info,cfg):
 originals={}
 for kind,col in [('state','observation.state'),('action','action')]:
  raw=arr(src,col);actual=arr(dst,kind+'_unified');mask=arr(dst,'mask_'+kind);names=info['features'][col]['names']
  assert actual.shape==(len(src),55) and mask.shape==actual.shape
  expected=np.zeros(actual.shape,dtype=np.float64);expected_mask=np.zeros(actual.shape,dtype=bool)
  def put(column,source_index,target_index):
   values=arr(src,column)[:,source_index];valid=np.isfinite(values)
   expected[:,target_index]=np.where(valid,values,0);expected_mask[:,target_index]=valid
  for side,base,gslot,hbase in [('left',0,28,30),('right',7,29,36)]:
   for part,offset,width in [('arm',base,7),('hand',hbase,6)]:
    pairs=sorted((int(m.group(1)),i) for i,n in enumerate(names) if (m:=re.fullmatch(side+'_'+part+r'_joint_(\d+)_rad',n)))
    if len(pairs)<=width:
     for number,index in pairs:put(col,index,offset+number-1)
   hits=[i for i,n in enumerate(names) if n in (side+'_gripper_open',side+'_gripper_open_scale')]
   if hits:put(col,hits[0],gslot)
   sim='eef_sim_pose_'+kind;sn=info['features'].get(sim,{}).get('names',[])
   wanted=[side+'_eef_pos_'+axis for axis in 'xyz']
   sdk=[side+'_end_pos_'+axis+'_m' for axis in 'xyz']+[side+'_end_quat_'+axis for axis in 'xyzw'];ebase=14+(0 if side=='left' else 7)
   if all(n in names for n in sdk):
    for j,n in enumerate(sdk[:3]):put(col,names.index(n),ebase+j)
    q=raw[:,[names.index(n) for n in sdk[3:]]];norm=np.linalg.norm(q,axis=1);good=np.isfinite(q).all(1)&(norm>1e-8)
    nq=np.zeros_like(q);nq[good]=q[good]/norm[good,None]
    rejected=[]
    for i in range(1,len(q)-1):
     if good[i-1:i+2].all():
      ang=lambda a,b:float(2*np.arccos(np.clip(abs(np.dot(a,b)),0,1))*180/np.pi)
      if ang(nq[i-1],nq[i])>120 and ang(nq[i],nq[i+1])>120 and ang(nq[i-1],nq[i+1])<30:rejected.append(i)
    good[rejected]=False
    expected_mask[:,ebase+3:ebase+7]=good[:,None];expected[good,ebase+3:ebase+7]=nq[good]
   elif all(n in sn for n in wanted):
    for axis,n in enumerate(wanted):put(sim,sn.index(n),ebase+axis)
  torso=[n for n in names if n.startswith('torso_joint_')]
  usable=sorted(n for n in torso if re.fullmatch(r'torso_joint_[1-4]_rad',n))
  if usable and len(usable)==len(torso):
   for j,n in enumerate(usable):put(col,names.index(n),42+j)
  waist=[n for n in ('waist_yaw_rad','waist_pitch_rad') if n in names]
  for j,n in enumerate(waist):put(col,names.index(n),42+j)
  for group in [('head_joint_1_rad','head_joint_2_rad'),('neck_joint_1_rad','neck_joint_2_rad'),('head_yaw_rad','head_pitch_rad')]:
   for j,n in enumerate(n for n in group if n in names):put(col,names.index(n),46+j)
  originals[kind]=(expected,expected_mask)
 from episode_policy import classify_episode
 s,sm=originals['state'];a,am=originals['action']
 decision=classify_episode(src,cfg,s,sm,a,am)
 derived=decision['mode']=='derived_next_state'
 if derived:
  a=np.concatenate([s[1:],np.zeros((1,55))]);am=np.concatenate([sm[1:],np.zeros((1,55),bool)])
 for kind,expected,expected_mask in [('state',s,sm),('action',a,am)]:
  actual=arr(dst,kind+'_unified');mask=arr(dst,'mask_'+kind)
  np.testing.assert_array_equal(mask,expected_mask)
  for base in [17,24]:
   valid=expected_mask[:,base:base+4].all(1)
   if valid.any():np.testing.assert_allclose(np.abs(np.sum(actual[valid,base:base+4]*expected[valid,base:base+4],axis=1)),1,atol=1e-6)
   expected[valid,base:base+4]=actual[valid,base:base+4]
  np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
 np.testing.assert_array_equal(arr(dst,'action_source_kind'),np.full(len(src),int(derived)))
 np.testing.assert_array_equal(arr(dst,'action_target_step'),np.r_[np.ones(len(src)-1),0] if derived else np.zeros(len(src)))
 times=arr(src,'timestamp');dt=np.r_[np.diff(times),0.];np.testing.assert_allclose(arr(dst,'sample_dt'),dt,atol=1e-8,rtol=1e-6)
 assert not arr(dst,'mask_source_dt').any(),'Published timestamps must not be labelled measured'

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());results=[]
 for job in plan['jobs']:
  unit=a.output/job['id'];record={'id':job['id'],'passed':False}
  try:
   receipt=json.loads((unit/'conversion_receipt.json').read_text());assert (unit/'SUCCESS.json').is_file()
   for x in receipt['inventory']:assert digest(unit/x['path'])==x['sha256'],x['path']
   manifest=json.loads((unit/'meta/input_manifest.json').read_text());info=json.loads((Path(job['source'])/'meta/info.json').read_text());cache=SourceCache();rows=0
   episodes=[]
   for q in sorted((unit/'meta/episodes').rglob('*.parquet')):episodes+=pq.read_table(q).to_pylist()
   assert len(episodes)==len(manifest['episodes'])
   for original,out in zip(manifest['episodes'],episodes):
    src=cache.episode(original['data'],Path('/'),original['length']);path=unit/f"data/chunk-{out['data/chunk_index']:03d}/file-{out['data/file_index']:03d}.parquet";dst=cache.get(path).slice(out['data/file_from_offset'],out['length']);assert len(src)==len(dst)
    for col in src.column_names:assert json.dumps(src[col].to_pylist(),sort_keys=True)==json.dumps(dst['source.'+col].to_pylist(),sort_keys=True),col
    check_mapping(src,dst,info,manifest);rows+=len(src)
   media=json.loads((unit/'meta/media_manifest.json').read_text());packet_count=0
   for f in media['files']:
    expected=packets(f['sources']);actual=packets([unit/f['path']]);assert expected==actual,'Compressed video payload/order changed';packet_count+=actual[0]
   ds=StandardDataset(unit)
   for index in sorted(set([0,len(ds)//2,len(ds)-1])):
    row=ds[index];assert row['state_unified'].shape==(55,) and row['action_unified'].shape==(55,)
   ds.close();record.update(passed=True,rows=rows,sha_files=len(receipt['inventory']),video_packets=packet_count)
  except Exception as e:record.update(error=str(e),traceback=traceback.format_exc())
  results.append(record);print(json.dumps(record),flush=True)
 passed=bool(results) and all(x['passed'] for x in results);report={'passed':passed,'corpus_complete':passed and not plan['skipped'] and not plan.get('missing_expected_repos',[]),'skipped':plan['skipped'],'missing_expected_repos':plan.get('missing_expected_repos',[]),'units':results};a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2));sys.exit(0 if passed else 1)
if __name__=='__main__':main()
