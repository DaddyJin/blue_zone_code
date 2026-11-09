"""Real offline LeRobot 0.5.0 reads, including every episode boundary and RGB comparison."""
from pathlib import Path
import os,sys,json,argparse,importlib.metadata
os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();P=Path(__file__).resolve().parents[1];sys.path=[x for x in sys.path if Path(x).resolve()!=Path(__file__).resolve().parent];sys.path.insert(0,str(P/'vendor/standard_format/tools'))
version=importlib.metadata.version('lerobot');assert version=='0.5.0',f'Requires lerobot 0.5.0, got {version}'
from lerobot_compat import load_dataset
from read_frame import StandardDataset
import numpy as np,pyarrow.parquet as pq
actual=load_dataset(a.root);reference=StandardDataset(a.root);assert len(actual)==len(reference)>0
indices={0,len(actual)//2,len(actual)-1}
for f in (a.root/'meta/episodes').rglob('*.parquet'):
 for e in pq.read_table(f).to_pylist():
  for i in (e['dataset_from_index'],e['dataset_to_index']-1):
   indices.add(i)
   if i>0:indices.add(i-1)
   if i+1<len(actual):indices.add(i+1)
images=0
for i in sorted(indices):
 x=actual[i];y=reference[i]
 for key in ['state_unified','action_unified','mask_state','mask_action']:
  assert tuple(x[key].shape)==(55,);np.testing.assert_array_equal(x[key].numpy(),y[key])
 for k,v in x.items():
  if k.startswith('observation.images.') and not k.endswith('.pts'):
   rgb=np.rint(v.numpy().transpose(1,2,0)*255).astype(np.uint8)
   np.testing.assert_array_equal(rgb,y[k]);images+=1
reference.close();print(json.dumps({'passed':True,'lerobot_version':version,'rows':len(actual),'indices':sorted(indices),'images_compared':images,'compatibility_subclass':True}))
