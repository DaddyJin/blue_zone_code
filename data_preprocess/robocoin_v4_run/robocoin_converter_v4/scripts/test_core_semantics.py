from pathlib import Path
import sys,json,copy
import numpy as np,pyarrow as pa
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'vendor/standard_format/tools'))
from common import map55
from quaternion_recovery import recover_quaternion
from episode_policy import apply_episode_policy
qs=np.array([[0,0,0,1],[0,0,0,-1],[0,0,0,.5],[0,0,0,1.02],[0,0,0,1e10],[0,0,0,0],[np.nan,0,0,1],[0,0,0,1e-12],[0,0,0,1e-4]])
q,g,r=recover_quaternion(qs);assert g.tolist()==[True]*5+[False]*3+[True];np.testing.assert_allclose(np.linalg.norm(q[g],axis=1),1)
q,g,r=recover_quaternion([[0,0,0,1],[1,0,0,.001],[0,0,0,1]]);assert g.tolist()==[True,False,True]
f=dict(source='pose',slot='left_eef',quantity='absolute_ee_pose',unit='m',provenance='measured',frame='native_sdk',reference_point='SDK end',quaternion_order='wxyz',quaternion_recovery='normalize_and_reject_isolated_spike')
x,m=map55(pa.table({'pose':[[1.,2.,3.,2.,0.,0.,0.],[1.,2.,3.,0.,0.,0.,0.]]}),[f]);assert x[0,20]==1 and m[0,17:21].all();assert m[:,14:17].all() and not m[1,17:21].any()
gf=dict(source='grip',slot='left_gripper',quantity='opening_directed_native_value',unit='source_native',provenance='measured',gripper={'mode':'native_open_direction','scale':1,'offset':0,'direction_evidence':'publisher'})
x,m=map55(pa.table({'grip':[-2.,0.,1.,1000.]}),[gf]);np.testing.assert_array_equal(x[:,28],[-2,0,1,1000]);assert not m[:,29].any()
names=['left_arm_joint_1_rad','right_arm_joint_1_rad','left_gripper_open']
sf=[dict(source='observation.state',indices=[i],slot=slot,unit='rad' if i<2 else 'source_native',component_names=[names[i]]) for i,slot in enumerate(['left_arm_joint','right_arm_joint','left_gripper'])]
af=[dict(f,source='action') for f in sf]
cfg={'state_fields':sf,'action':{'fields':af,'copy_detection_names':{'state':names,'action':names}}}
checks=[]
def run(name,raw,action,expected):
 t=pa.table({'observation.state':raw.tolist(),'action':action.tolist()});s=np.zeros((len(raw),55));a=s.copy();ms=s.copy();ma=s.copy()
 for i,j in enumerate([0,7,28]):s[:,j]=raw[:,i];a[:,j]=action[:,i];ms[:,j]=1;ma[:,j]=1
 out,mask,r=apply_episode_policy(t,cfg,s,ms,a,ma);assert r['mode']==expected,(name,r)
 if expected=='derived_next_state':
  np.testing.assert_array_equal(out[:-1],s[1:]);assert not out[-1].any() and not mask[-1].any()
 else:np.testing.assert_array_equal(out,a)
 checks.append(name)
r=np.array([[0.,0.,1.],[.1,.2,.8],[.2,.3,.5],[.3,.4,.2]])
run('dynamic_exact_copy',r,r.copy(),'derived_next_state')
run('static_ambiguous',np.ones((4,3)),np.ones((4,3)),'source_native')
run('already_next',r,np.vstack([r[1:],r[-1:]]),'source_native')
a=r.copy();a[2,2]=0;run('native_grasp_command',r,a,'source_native')
a=r.copy();a[-1,0]+=1e-8;run('single_row_difference',r,a,'source_native')
run('too_short',r[:1],r[:1],'source_native')
# Sequential calls do not carry policy across episode boundaries.
run('mixed_repo_copy_episode',r,r,'derived_next_state');run('mixed_repo_native_episode',r,a,'source_native')
print(json.dumps({'passed':True,'action_cases':checks,'quaternion_checks':['normalization beyond 1pct','tiny and nonfinite rejected','isolated spike','wxyz reorder','xyz survives invalid rotation'],'gripper_native_values_preserved':True}))
