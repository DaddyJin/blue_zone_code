from pathlib import Path
import sys
import numpy as np,pyarrow.parquet as pq,json,re,collections,argparse
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'vendor/standard_format/tools'))
from quaternion_recovery import recover_quaternion
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=P/'research/semantic_audit_samples');parser.add_argument('--report',type=Path,default=P/'reports/semantic_audit_809.json');parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();ROOT=args.input
def relation(x,z):
 ok=np.isfinite(x)&np.isfinite(z);x=x[ok];z=z[ok];r={'n':len(x)}
 if not len(x):return dict(r,status='no_finite_pairs')
 r.update(raw_range=[float(x.min()),float(x.max())],scale_range=[float(z.min()),float(z.max())])
 if np.ptp(x)<1e-7 or np.ptp(z)<1e-5:return dict(r,status='uninformative_constant')
 order=np.argsort(x,kind='stable');sx=x[order];sz=z[order];positive=bool(np.all(np.diff(sz)>=-1e-4));negative=bool(np.all(np.diff(sz)<=1e-4));corr=float(np.corrcoef(x,z)[0,1]);r['corr']=corr
 r['status']='increasing' if positive and corr>0.1 else ('decreasing' if negative and corr<-.1 else 'conflict_or_nonlinear')
 return r

def audit(root):
 info_path=root/'meta/info.json' if (root/'meta/info.json').exists() else root/'info.json';info=json.loads(info_path.read_text());assert info['codebase_version']=='v2.1';result={'repo':root.name,'robot_type':info.get('robot_type'),'has_gripper':'left_gripper_open' in info['features']['observation.state']['names'],'episodes':[]}
 files=sorted((root/'data').rglob('*.parquet')) if (root/'data').exists() else sorted(root.glob('*.parquet'))
 if not files:raise ValueError('No local Parquet files found')
 for f in files:
  try:
   t=pq.read_table(f,use_threads=False);row={'episode':int(re.search(r'(\d+)$',f.stem).group(1)),'rows':len(t),'grippers':{},'joint_lags':[]};state=np.array(t['observation.state'].to_pylist(),dtype=float);action=np.array(t['action'].to_pylist(),dtype=float);ns=info['features']['observation.state']['names'];na=info['features']['action']['names']
   for kind,col,names,arr in [('state','observation.state',ns,state),('action','action',na,action)]:
    for side,j in [('left',0),('right',1)]:
     name=side+'_gripper_open'
     if name not in names:continue
     scale='gripper_open_scale_'+kind;x=arr[:,names.index(name)]
     row['grippers'][kind+'_'+side]=relation(x,np.array(t[scale].to_pylist(),dtype=float)[:,j]) if scale in t.column_names else {'status':'missing_scale'}
   row['sdk_quaternions']={}
   for kind,names_q,arr_q in [('state',ns,state),('action',na,action)]:
    for side in ['left','right']:
     qnames=[side+'_end_quat_'+axis for axis in 'xyzw']
     if all(n in names_q for n in qnames):
      q=arr_q[:,[names_q.index(n) for n in qnames]];norm=np.linalg.norm(q,axis=1);finite=np.isfinite(q).all(1);valid=finite&(np.abs(norm-1)<=.01)
      row['sdk_quaternions'][kind+'_'+side]={'order':'xyzw','rows':len(q),'valid_within_1pct_norm':int(valid.sum()),'outside_1pct_norm_diagnostic_only':int((~valid).sum()),'v4_recovery':recover_quaternion(q)[2],'min_finite_norm':float(norm[finite].min()) if finite.any() else None,'max_finite_norm':float(norm[finite].max()) if finite.any() else None}
   names=[n for n in ns if re.fullmatch(r'(left|right)_arm_joint_\d+_rad',n) and n in na];s=state[:,[ns.index(n) for n in names]];a=action[:,[na.index(n) for n in names]]
   row['dynamic_joint_entries']=int(np.sum(np.abs(np.diff(s,axis=0))>1e-5));row['joint_dims']=len(names)
   core_names=names+[n for n in ('left_gripper_open','right_gripper_open') if n in ns and n in na]
   core_s=state[:,[ns.index(n) for n in core_names]];core_a=action[:,[na.index(n) for n in core_names]]
   core_lags=[]
   for lag in [-2,-1,0,1,2]:
    x,y=(core_a[:-lag],core_s[lag:]) if lag>0 else ((core_a[-lag:],core_s[:lag]) if lag<0 else (core_a,core_s))
    if len(x) and np.all(np.abs(x-y)<=1e-6):core_lags.append(lag)
   row['core_exact_lags_1e6']=core_lags
   row['core_time_relation']='same_row' if core_lags==[0] else ('next_state' if core_lags==[1] else ('other_or_not_equal' if not core_lags else 'ambiguous_static_or_multiple'))

   for lag in [-2,-1,0,1,2]:
    x,y=(a[:-lag],s[lag:]) if lag>0 else ((a[-lag:],s[:lag]) if lag<0 else (a,s))
    if not len(x):continue
    e=np.abs(x-y);row['joint_lags'].append({'lag_state_minus_action':lag,'mae':float(np.mean(e)) if np.isfinite(e).all() else None,'max_abs':float(np.max(e)) if np.isfinite(e).all() else None,'exact':bool(np.array_equal(x,y)),'allclose_1e6':bool(np.all(e<=1e-6))})
   valid=[v['lag_state_minus_action'] for v in row['joint_lags'] if v['allclose_1e6']]
   row['time_relation']='same_row' if valid==[0] else ('next_state' if valid==[1] else ('other_or_not_equal' if not valid else 'ambiguous_static_or_multiple'))
   row['exact_lags']=valid;row['nonfinite_joint_entries']=int(np.sum(~np.isfinite(s))+np.sum(~np.isfinite(a)));result['episodes'].append(row)
  except Exception as e:result['episodes'].append({'episode':int(re.search(r'(\d+)$',f.stem).group(1)),'error':str(e)})
 result['time_relation']=next(iter({r.get('time_relation','error') for r in result['episodes']})) if len({r.get('time_relation','error') for r in result['episodes']})==1 else 'mixed'
 result['core_time_relation']=next(iter({r.get('core_time_relation','error') for r in result['episodes']})) if len({r.get('core_time_relation','error') for r in result['episodes']})==1 else 'mixed'
 result['gripper_lanes']={}
 if result['has_gripper']:
  for lane in ['state_left','state_right','action_left','action_right']:
   statuses={r.get('grippers',{}).get(lane,{}).get('status','error') for r in result['episodes']}
   informative=statuses-{'uninformative_constant'}
   result['gripper_lanes'][lane]=next(iter(informative)) if len(informative)==1 else ('uninformative_constant' if not informative else 'mixed:'+','.join(sorted(informative)))
 return result
def audit_safe(root):
 try:return audit(root)
 except Exception as e:return {'repo':root.name,'robot_type':None,'has_gripper':False,'time_relation':'error','core_time_relation':'error','gripper_lanes':{},'episodes':[{'error':str(e)}]}
roots=([ROOT] if (ROOT/'meta/info.json').exists() else sorted({q.parent.parent for q in ROOT.rglob('meta/info.json')} | {q.parent for q in ROOT.glob('*/info.json')}))
assert roots,'No source repositories found'
with ThreadPoolExecutor(args.workers) as pool:rows=list(pool.map(audit_safe,roots))
c=collections.Counter(r['time_relation'] for r in rows);g=collections.Counter(v for r in rows for v in r['gripper_lanes'].values());report={'scope':'All local Parquet files under the supplied input; default audit fixtures contain first/last episodes only, not the full corpus','repositories':rows,'time_repository_counts':dict(c),'core_time_repository_counts':dict(collections.Counter(r['core_time_relation'] for r in rows)),'gripper_lane_counts':dict(g),'total_rows':sum(ep.get('rows',0) for r in rows for ep in r['episodes']),'errors':[{'repo':r['repo'],**e} for r in rows for e in r['episodes'] if 'error' in e]}
args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='repositories'},indent=2))
