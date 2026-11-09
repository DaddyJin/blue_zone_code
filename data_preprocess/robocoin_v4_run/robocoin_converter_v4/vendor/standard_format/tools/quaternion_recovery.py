"""Normalize known-order quaternions, without fabricating missing orientation."""
import numpy as np

def recover_quaternion(raw):
    raw=np.asarray(raw,dtype=np.float64)
    scale=np.max(np.abs(raw),axis=1)
    finite=np.isfinite(raw).all(1)
    q=np.zeros_like(raw);good=finite&(scale>0)
    q[good]=raw[good]/scale[good,None]
    scaled_norm=np.linalg.norm(q,axis=1)
    # Equivalent norm >1e-8, with stable normalization for large finite inputs.
    good &= scale>np.divide(1e-8,scaled_norm,out=np.full(len(q),np.inf),where=scaled_norm>0)
    q[good]/=scaled_norm[good,None];q[~good]=0
    original_good=good.copy();spikes=np.zeros(len(q),bool)
    if len(q)>2:
        def angle(x,y):return np.degrees(2*np.arccos(np.clip(np.abs(np.sum(x*y,axis=1)),0,1)))
        prev=angle(q[1:-1],q[:-2]);nxt=angle(q[1:-1],q[2:]);bridge=angle(q[:-2],q[2:])
        spikes[1:-1]=good[:-2]&good[1:-1]&good[2:]&(prev>120)&(nxt>120)&(bridge<30)
    good &= ~spikes;q[~good]=0
    for i in range(1,len(q)):
        if good[i] and good[i-1] and np.dot(q[i],q[i-1])<0:q[i]*=-1
    with np.errstate(over='ignore',invalid='ignore'):
        norm=scale*scaled_norm
    old_good=finite&(np.abs(norm-1)<=.01)
    return q,good,{'total':len(q),'valid':int(good.sum()),'recovered_beyond_old_1pct':int((good&~old_good).sum()),
                  'invalid_or_near_zero':int((~original_good).sum()),'isolated_spikes':int(spikes.sum()),
                  'policy':'finite norm>1e-8; normalize; reject isolated >120deg jumps with neighbours <30deg apart; heuristic QC, not calibrated robot limit'}
