"""Explicit, lossless name completion for audited RoboCOIN schemas; never trim data."""
from pathlib import Path
import copy,hashlib,json
P=Path(__file__).resolve().parents[1]
def key(info):
 return hashlib.sha256(json.dumps({k:v for k,v in info['features'].items() if k in ('observation.state','action','eef_sim_pose_state','eef_sim_pose_action')},sort_keys=True).encode()).hexdigest()
def rule_for(source,info):
 rules=json.loads((P/'research/dimension_repairs.json').read_text())['rules']
 for r in rules:
  if key(info)==r['schema_key'] and Path(source).name in r['repositories']:return r
 return None
def repair_info(source,info):
 rule=rule_for(source,info)
 if rule is None:return info,None
 fixed=copy.deepcopy(info);changes=[]
 for col,spec in rule['columns'].items():
  f=fixed['features'][col];n=len(f['names']);width=f['shape'][0]
  if n!=spec['named_width'] or width!=spec['actual_width']:raise ValueError('Repair rule no longer matches source')
  extra=[f'unnamed_source_component_{i}' for i in range(n,width)]
  f['names']=f['names']+extra
  changes.append({'column':col,'original_names':info['features'][col]['names'],'declared_and_actual_width':width,'unknown_indices':list(range(n,width)),'effective_names':f['names']})
 return fixed,{'rule_id':rule['id'],'scope':'Name completion only; unknown channels retained without inferred physical semantics','evidence':rule['evidence'],'changes':changes}
def validate_vectors(source,info):
 """Scan ALL local parquet rows before allowing a repair, including non-sampled episodes."""
 import pyarrow as pa,pyarrow.parquet as pq,pyarrow.compute as pc
 fixed,repair=repair_info(source,info)
 if repair is None:return None
 records=[]
 for f in sorted((Path(source)/'data').rglob('*.parquet')):
  pf=pq.ParquetFile(f);counts={k:0 for k in ['observation.state','action']};nonfinite=dict(counts)
  for batch in pf.iter_batches(batch_size=8192,columns=list(counts)):
   for col in counts:
    a=batch.column(batch.schema.get_field_index(col));width=info['features'][col]['shape'][0]
    if not (pa.types.is_list(a.type) or pa.types.is_large_list(a.type) or pa.types.is_fixed_size_list(a.type)):raise ValueError(f'{f}:{col}: expected vector, got {a.type}')
    if a.null_count:raise ValueError(f'{f}:{col}: null vector rows cannot be repaired with names')
    lengths=pc.list_value_length(a)
    if not pc.all(pc.equal(lengths,width)).as_py():raise ValueError(f'{f}:{col}: actual vector widths differ from {width}; refused metadata-only repair')
    flat=pc.list_flatten(a)
    if not pa.types.is_floating(flat.type):raise ValueError(f'{f}:{col}: unexpected scalar dtype {flat.type}')
    counts[col]+=len(a);nonfinite[col]+=int(pc.sum(pc.invert(pc.fill_null(pc.is_finite(flat),False))).as_py() or 0)
  records.append({'file':str(f.relative_to(source)),'rows':pf.metadata.num_rows,'vector_rows_scanned':counts,'nonfinite_values':nonfinite})
 if not records:raise ValueError('No parquet files available to validate metadata repair')
 return dict(repair,validation={'mode':'all_local_parquet_rows','files':records,'total_rows':sum(x['rows'] for x in records)})
