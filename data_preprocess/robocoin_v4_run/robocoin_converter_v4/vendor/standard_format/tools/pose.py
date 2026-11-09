"""Optional rigid transform for ABSOLUTE poses only. Never infers calibration."""
import numpy as np
from scipy.spatial.transform import Rotation

def transform_absolute_pose(pose,T_target_from_source,source_order='xyzw'):
    p=np.asarray(pose,np.float64)
    if p.ndim!=2 or p.shape[1]!=7:raise ValueError('Expected Nx7 xyz+quaternion')
    if source_order not in ('xyzw','wxyz'):raise ValueError('Explicit quaternion order required')
    T=np.asarray(T_target_from_source,np.float64)
    if T.shape==(4,4):T=np.broadcast_to(T,(len(p),4,4))
    if T.shape!=(len(p),4,4) or not np.isfinite(T).all():raise ValueError('Expected finite 4x4 or Nx4x4 rigid transforms')
    R=T[:,:3,:3]
    if not np.allclose(T[:,3,:],[0,0,0,1],atol=1e-6) or not np.allclose(R.transpose(0,2,1)@R,np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(R),1,atol=1e-5):raise ValueError('Transform is not a proper rigid transform')
    q=p[:,3:7] if source_order=='xyzw' else p[:,[4,5,6,3]]
    valid=np.isfinite(p).all(1)&(np.linalg.norm(q,axis=1)>1e-8);out=np.zeros_like(p)
    out[valid,:3]=np.einsum('nij,nj->ni',R[valid],p[valid,:3])+T[valid,:3,3]
    out[valid,3:7]=(Rotation.from_matrix(R[valid])*Rotation.from_quat(q[valid])).as_quat()
    return out.astype(np.float32),valid
