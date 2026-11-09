"""Conservative, full-episode action policy. No inferred command latency correction."""
import numpy as np

def classify_episode(table, config, state, ms, action, ma):
    report={'mode':'source_native','reason':'not_confirmed_copy','rows':len(table),
            'comparison':'exact original values over every mapped field and shared named raw component; entire episode',
            'command_provenance':'publisher supplied; command-vs-derived provenance not established by this label'}
    if len(table)<3:
        return dict(report,reason='too_short')
    sf={f['slot']:f for f in config['state_fields']};af={f['slot']:f for f in config['action']['fields']}
    if set(sf)!=set(af):return dict(report,reason='different_mapped_fields')
    # Require every mapped value to be valid; invalid/unknown rotations cannot prove a copy.
    for slot,f in sf.items():
        g=af[slot]
        for key in ('components','component_names','unit','source_unit','frame','quaternion_order'):
            if f.get(key)!=g.get(key):return dict(report,reason='different_field_semantics:'+slot)
        x=np.asarray(table[f['source']].to_pylist(),dtype=np.float64)
        y=np.asarray(table[g['source']].to_pylist(),dtype=np.float64)
        if x.ndim==1:x=x[:,None]
        if y.ndim==1:y=y[:,None]
        x=x[:,f['indices']];y=y[:,g['indices']]
        if not np.isfinite(x).all() or not np.isfinite(y).all():return dict(report,reason='nonfinite:'+slot)
        if not np.array_equal(x,y):return dict(report,reason='source_values_differ:'+slot)
    names=config.get('action',{}).get('copy_detection_names',{})
    sn,an=names.get('state',[]),names.get('action',[])
    shared=[n for n in sn if n in an]
    if shared:
        x=np.asarray(table['observation.state'].to_pylist(),dtype=np.float64)[:,[sn.index(n) for n in shared]]
        y=np.asarray(table['action'].to_pylist(),dtype=np.float64)[:,[an.index(n) for n in shared]]
        if not np.isfinite(x).all() or not np.array_equal(x,y):return dict(report,reason='shared_raw_components_differ')
    core=np.r_[0:14,28:30];used=(ms[:,core]>0).any(0)
    if not used.any() or not (ms[:,core][:,used]>0).all() or not np.array_equal(ms,ma):
        return dict(report,reason='invalid_or_different_masks')
    if not np.any(np.abs(np.diff(state[:,core][:,used].astype(np.float64),axis=0))>1e-6):
        return dict(report,reason='static_ambiguous')
    return dict(report,mode='derived_next_state',reason='dynamic_full_episode_exact_copy',
                terminal='all action values and masks zero; no next frame',source_step=1)

def apply_episode_policy(table,config,state,ms,action,ma):
    report=classify_episode(table,config,state,ms,action,ma)
    if report['mode']=='derived_next_state':
        action=np.zeros_like(state);ma=np.zeros_like(ms)
        action[:-1]=state[1:];ma[:-1]=ms[1:]
    return action,ma,report
