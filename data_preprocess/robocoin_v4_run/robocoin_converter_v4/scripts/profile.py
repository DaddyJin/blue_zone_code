"""Evidence-based RoboCOIN named-component mapping; originals always retained."""
import re,hashlib,json
EVIDENCE='https://github.com/FlagOpen/RoboCOIN#lerobot-features-explanation'
def schema_key(info):
 return hashlib.sha256(json.dumps({k:v for k,v in info['features'].items() if k in ('observation.state','action','eef_sim_pose_state','eef_sim_pose_action')},sort_keys=True).encode()).hexdigest()
def profile(source,info,split):
 from repair_schema import repair_info
 from repository_catalog import identify
 repo=identify(source)
 info,repair=repair_info(source,info)
 robot=info.get('robot_type') or 'publisher_robot_type_unspecified:'+source.name
 mapped={};frames={};notes=[]
 for kind,col in [('state','observation.state'),('action','action')]:
  fs=[];feature=info['features'].get(col);names=feature.get('names') if feature else None
  if not isinstance(names,list) or len(names)!=feature['shape'][0] or len(set(names))!=len(names):raise ValueError('Missing/ambiguous component names: '+col)
  def add(slot,ix,quantity='joint_position',unit='rad',**kw):
   fs.append(dict(source=col,indices=ix,slot=slot,owner=robot,quantity=quantity,unit=unit,component_names=[names[i] for i in ix],provenance='measured' if kind=='state' else 'derived',**({'method':'Publisher action: master/slave command or state-derived target; per-source distinction not published'} if kind=='action' else {}),**kw))
  for side in ['left','right']:
   for part,slot,maxn in [('arm',side+'_arm_joint',7),('hand',side+'_hand',6)]:
    pairs=sorted((int(m.group(1)),i) for i,n in enumerate(names) if (m:=re.fullmatch(side+'_'+part+r'_joint_(\d+)_rad',n)))
    if pairs:
     if [x[0] for x in pairs]!=list(range(1,len(pairs)+1)):raise ValueError('Noncontiguous joint names')
     if len(pairs)<=maxn:add(slot,[x[1] for x in pairs])
     else:notes.append(f'{kind}:{side}_{part}: {len(pairs)} channels exceed slot width {maxn}; all retained in source, no truncation')
   hits=[i for i,n in enumerate(names) if n in (side+'_gripper_open',side+'_gripper_open_scale')]
   if len(hits)>1:raise ValueError('Ambiguous gripper')
   if hits:add(side+'_gripper',hits,'opening_directed_native_value','source_native',gripper={'mode':'native_open_direction','scale':1,'offset':0,'direction_evidence':'https://github.com/FlagOpen/RoboCOIN/pull/7/files ; publisher originally defined gripper_open as 0 closed / 1 open; native scale retained; per-repository numeric checks and annotation conflicts documented in reports/semantic_audit_809.json'})
  # Only well-defined contiguous waist channels fit; do not truncate 5-axis torsos.
  ix=[i for i,n in enumerate(names) if re.fullmatch(r'torso_joint_[1-4]_rad',n)]
  alltorso=[n for n in names if n.startswith('torso_joint_')]
  if ix and len(ix)==len(alltorso):add('waist',sorted(ix,key=lambda i:names[i]))
  waist_named=[names.index(n) for n in ('waist_yaw_rad','waist_pitch_rad') if n in names]
  if waist_named:
   if ix:raise ValueError('Ambiguous waist components')
   add('waist',waist_named)
  head_groups=[[names.index(n) for n in group if n in names] for group in [('head_joint_1_rad','head_joint_2_rad'),('neck_joint_1_rad','neck_joint_2_rad'),('head_yaw_rad','head_pitch_rad')]]
  head_groups=[x for x in head_groups if x]
  if len(head_groups)>1:raise ValueError('Ambiguous head/neck components')
  if head_groups:
   frame='joint_local_'+robot;frames[frame]={'kind':'joint_local','axes':'Per-joint rotation axes; geometric directions unspecified','origin':'Per-robot head/neck joint zero; geometric axes not retargeted','status':'publisher_named_joint','evidence':EVIDENCE}
   add('head',head_groups[0],frame=frame)
  for side in ['left','right']:
   arm=[f for f in fs if f['slot']==side+'_arm_joint']
   if len(arm)!=1 or len(arm[0]['indices']) not in (6,7):raise ValueError('Required arm joint representation missing/unsupported: '+side)
  # SDK Euler convention and SDK frames are not inferred. Keep original complete poses.
  sim='eef_sim_pose_'+kind;sf=info['features'].get(sim,{})
  sn=sf.get('names',[])
  for side in ['left','right']:
   sdk_names=[side+'_end_pos_'+a+'_m' for a in 'xyz']+[side+'_end_quat_'+a for a in 'xyzw']
   if all(n in names for n in sdk_names):
    frame='robocoin_sdk_'+robot;frames[frame]={'kind':'publisher_sdk_frame','origin':'Native published SDK frame; exact origin not published; no cross-robot frame alignment claimed','axes':'Native SDK axes; no camera/world transform inferred','status':'native_frame_preserved','evidence':'Published meta/info.json component names: xyz meters and explicit quaternion xyzw'}
    fs.append(dict(source=col,indices=[names.index(n) for n in sdk_names],slot=side+'_eef',owner=robot,quantity='absolute_ee_pose',unit='m',frame=frame,reference_point=side+' original SDK end pose (exact physical link not published)',quaternion_order='xyzw',quaternion_recovery='normalize_and_reject_isolated_spike',provenance='measured' if kind=='state' else 'derived',**({'method':'Published native SDK action pose; source row preserved'} if kind=='action' else {})))
    continue
   wanted=[side+'_eef_pos_'+a for a in 'xyz']
   if all(x in sn for x in wanted):
    frame='robocoin_sim_'+robot;frames[frame]={'kind':'publisher_simulation_frame','origin':'Publisher specifies robot base or centre between feet; exact per-model URDF origin is not provided','axes':'x forward, y left, z up','status':'publisher_declared','evidence':EVIDENCE}
    fs.append(dict(source=sim,indices=[sn.index(x) for x in wanted],components=[0,1,2],slot=side+'_eef',owner=robot,quantity='absolute_ee_position',unit='m',frame=frame,reference_point=side+' publisher simulated end effector (exact link unspecified)',provenance='derived',method='Publisher simulation FK; original Euler values retained separately; quaternion convention not inferred'))
  used={i for f in fs if f['source']==col for i in f['indices']}
  notes.append({'kind':kind,'unmapped_source_components':[n for i,n in enumerate(names) if i not in used]})
  mapped[kind]=fs
 return dict(dataset_id=repo['canonical_name']+'__'+split,profile_id='robocoin:named-components:v4:'+schema_key(info),robot_type=robot,data_kind='robot',fps=info['fps'],split=split,source={'id':repo['canonical_repo'],'revision':'local-info-sha256:'+hashlib.sha256((source/'meta/info.json').read_bytes()).hexdigest(),'dataset_url':'https://modelscope.cn/datasets/'+repo['modelscope_repo']},state_fields=mapped['state'],action={'mode':'source','terminal':'keep','fields':mapped['action'],'episode_copy_policy':'exact_dynamic_full_episode','copy_detection_names':{'state':info['features']['observation.state']['names'],'action':info['features']['action']['names']}},frames=frames,media={'policy':'native','max_file_bytes':128_000_000},time={'kind':'published','column':'timestamp','unit':'s'},passthrough_columns='all',semantic_notes={'core_policy':{'required':'both arm joint positions in rad','dex_hand_gripper':'empty mask; all finger joints retained in source columns','action_alignment':'preserve published rows except exact dynamic full-episode state copies; decision saved per episode and row','audit_scope':'809 repository first/last episode audit, not full corpus'},'repository':repo,'evidence':EVIDENCE,'unmapped':notes,'time':'Published timestamps preserved; not proven measured acquisition time','poses':'Explicit native SDK xyz+xyzw mapped as a complete pair where available (30 catalog repositories). Finite nonzero SDK quaternions normalized regardless of norm deviation, with isolated >120 degree reversal spikes rejected and episode-local sign continuity; no invented interpolation. Otherwise only simulation xyz is mapped; published simulation Euler generation convention has not been located in public producer code.','metadata_repair':repair,'gripper':'Preserve native numeric scale, larger means more open per publisher component naming and documented direction; do not clip or assume normalization endpoints'})
