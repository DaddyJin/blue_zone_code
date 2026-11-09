"""Small explicit adapter utility; loads numeric/string 1D/2D NPZ arrays only."""
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow as pa
from common import write_table
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--keys',nargs='+');a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    columns={};length=None
    with np.load(a.source,allow_pickle=False) as archive:
        for key in a.keys or archive.files:
            x=archive[key]
            if x.ndim not in (1,2) or x.dtype.kind not in 'biufUS':raise ValueError(f'{key}: adapt objects/nested tensors explicitly')
            if x.dtype.kind=='S':x=np.char.decode(x,'utf-8')
            if length is not None and len(x)!=length:raise ValueError('Arrays have different numbers of samples')
            length=len(x);columns[key]=pa.array(x.tolist())
    write_table(a.output,pa.table(columns));print(json.dumps({'output':a.output,'rows':length,'columns':list(columns)}))
