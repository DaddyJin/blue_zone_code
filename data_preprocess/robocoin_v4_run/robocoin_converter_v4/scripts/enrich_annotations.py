"""Use the publisher's ordered concatenation of subtask labels, without inventing hierarchy."""
import json
from pathlib import Path
import pyarrow.parquet as pq

def enrich(source,manifest):
 p=source/'annotations/subtask_annotations.jsonl'
 c=json.loads(manifest.read_text())
 if not p.is_file():
  c['semantic_notes']['subtasks']='Dictionary unavailable; original annotation IDs retained, no text inferred'
 else:
  words={int(r['subtask_index']):r['subtask'] for line in p.read_text().splitlines() if line.strip() for r in [json.loads(line)]}
  for e in c['episodes']:
   labels=[]
   for ref in e['data']:
    pf=pq.ParquetFile(ref['path'])
    if 'subtask_annotation' not in pf.schema_arrow.names:labels+=['']*ref['length'];continue
    values=pf.read(columns=['subtask_annotation']).slice(ref['offset'],ref['length'])['subtask_annotation'].to_pylist()
    for ids in values:
     ids=ids if isinstance(ids,list) else [ids]
     # Missing IDs are corrupt annotation metadata, not a reason to invent labels.
     if any(int(i) not in words for i in ids):raise ValueError('Unknown subtask annotation ID')
     labels.append(' '.join(words[int(i)] for i in ids if words[int(i)]!='null'))
   segments=[];start=0
   for end in range(1,len(labels)+1):
    if end==len(labels) or labels[end]!=labels[start]:
     if labels[start]:segments.append({'start_frame':start,'end_frame':end,'action_text':labels[start],'skill':None})
     start=end
   e['segments']=segments
  c['semantic_notes']['subtasks']='Ordered join excluding literal null, matching upstream LeRobot reader; original IDs and dictionaries retained'
 manifest.write_text(json.dumps(c,ensure_ascii=False,indent=2))
