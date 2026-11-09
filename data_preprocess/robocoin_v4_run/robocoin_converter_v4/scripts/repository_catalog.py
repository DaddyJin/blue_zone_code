"""Resolve published HF / ModelScope aliases without treating their revisions as identical."""
import json
from functools import lru_cache
from pathlib import Path
P=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=1)
def catalog():
 rows=json.loads((P/'research/repository_aliases.json').read_text())['repos'];by_name={}
 for row in rows:
  for key in ('canonical_repo','modelscope_repo'):
   name=row[key].split('/')[-1]
   assert name not in by_name or by_name[name]['canonical_repo']==row['canonical_repo'],'Ambiguous repository alias'
   by_name[name]=row
 return by_name

def identify(source):
 # The extracted dataset directory must retain its published repository name.
 name=Path(source).name;row=catalog().get(name)
 if row is None:raise ValueError('Unknown repository directory name: '+name+'; retain its HF or ModelScope repository name')
 return dict(row,local_directory_name=name,canonical_name=row['canonical_repo'].split('/')[-1],
             revision_policy='Use actual local metadata and files; HF and ModelScope revisions may differ')
