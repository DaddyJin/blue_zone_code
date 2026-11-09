"""Local extracted RoboCOIN conversion. No downloading or extraction."""
from pathlib import Path
import argparse,sys,os,json,hashlib,time,shutil,fcntl,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'vendor/standard_format/tools'))
from profile import profile,schema_key
from repair_schema import rule_for,validate_vectors,repair_info
from repository_catalog import identify

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def input_files(root):
 files={}
 for start in [root]+[root/x for x in ("data","videos","meta","annotations") if (root/x).is_dir()]:
  for p in start.rglob("*"):
   if p.is_file():files[str(p)]=p
 return sorted(files.values())
def source_fingerprint(root):
 h=hashlib.sha256()
 for p in input_files(root):
  if p.is_file():
   st=p.stat();h.update(json.dumps([str(p.relative_to(root)),st.st_size,st.st_mtime_ns]).encode())
   if p.suffix in ('.json','.jsonl','.yaml'):h.update(p.read_bytes())
 return h.hexdigest()
def job_run(j,out,work,vworkers,mb,dthreads):
 start=time.monotonic();source=Path(j['source']);wd=work/j['id'];wd.mkdir(parents=True,exist_ok=True);target=out/j['id']
 info=json.loads((source/'meta/info.json').read_text());cfg=profile(source,info,j['split']);cfg['source_split_ranges']=j['ranges'];cfg['media']['max_file_bytes']=mb*1000000
 code=hashlib.sha256()
 for p in [P/'scripts'/n for n in ['convert_repos.py','profile.py','repair_schema.py','repository_catalog.py','enrich_annotations.py','fast_media.py']]+sorted((P/'vendor/standard_format/tools').glob('*.py')):code.update(p.name.encode()+p.read_bytes())
 code.update((P/'research/dimension_repairs.json').read_bytes())
 code.update((P/'research/repository_aliases.json').read_bytes())
 signature=hashlib.sha256((j['source_fingerprint']+json.dumps(cfg,sort_keys=True)+code.hexdigest()+str(dthreads)).encode()).hexdigest()
 receipt=target/'conversion_receipt.json'
 if target.exists():
  r=json.loads(receipt.read_text());assert r['signature']==signature,'Existing output differs; choose new output version'
  assert all((target/x['path']).is_file() and digest(target/x['path'])==x['sha256'] for x in r['inventory']),'Output hash mismatch'
  return {'id':j['id'],'status':'skipped_verified','seconds':time.monotonic()-start}
 final_target=target;target=out/('.'+j['id']+'.pending')
 if target.exists():
  assert not target.is_symlink(),'Refuse symlink in staging area'
  shutil.rmtree(target)
 receipt=target/'conversion_receipt.json'
 if j.get('metadata_repair'):cfg['semantic_notes']['metadata_repair']=j['metadata_repair']
 save(wd/'profile.json',cfg)
 from from_lerobot import adapt
 adapt(source,wd/'profile.json',wd/'manifest.json')
 from enrich_annotations import enrich
 enrich(source,wd/'manifest.json')
 import fast_media
 fast_media.install();os.environ['GM100_VIDEO_WORKERS']=str(vworkers);os.environ['GM100_LINK_MODE']='copy';os.environ['ROBOCOIN_DECODE_THREADS']=str(dthreads)
 import pyarrow as pa
 pa.set_cpu_count(1);pa.set_io_thread_count(1)
 from convert import convert
 convert(wd/'manifest.json',target)
 shutil.copytree(source/'meta',target/'meta/source_metadata')
 if j.get('metadata_repair'):
  save(target/'meta/metadata_repair.json',j['metadata_repair'])
  save(target/'meta/source_info_effective.json',repair_info(source,info)[0])
 if (source/'annotations').exists():shutil.copytree(source/'annotations',target/'meta/source_annotations')
 for name in ['README.md','info.yaml']:
  if (source/name).is_file():shutil.copy2(source/name,target/'meta/source_metadata'/name)
 assert source_fingerprint(source)==j['source_fingerprint'],'Source changed during conversion'
 inv=[{'path':str(p.relative_to(target)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(target.rglob('*')) if p.is_file()]
 save(receipt,{'signature':signature,'inventory':inv,'source':str(source),'repository':identify(source)})
 target.rename(final_target)
 return {'id':j['id'],'status':'complete','seconds':time.monotonic()-start}
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','convert']);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--only-repairable',action='store_true',help='Select the 54 name-incomplete repositories only, for incremental recovery');p.add_argument('--workers',type=int,default=16);p.add_argument('--video-workers',type=int,default=4);p.add_argument('--decode-threads',type=int,default=1);p.add_argument('--cpu-budget',type=int,default=128);p.add_argument('--video-mb',type=int,default=128);a=p.parse_args()
 inp,out,work=[x.resolve() for x in (a.input,a.output,a.work)]
 assert all(not x.is_relative_to(y) and not y.is_relative_to(x) for x,y in [(inp,out),(inp,work),(out,work)]),'Directories must be disjoint'
 assert min(a.workers,a.video_workers,a.video_mb,a.decode_threads,a.cpu_budget)>0
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:a.cpu_budget])
 assert a.workers*a.video_workers*a.decode_threads<=len(os.sched_getaffinity(0)),'Requested parallelism exceeds CPU budget'
 out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True);lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 known=set(json.loads((P/'research/validated_schemas.json').read_text())['schemas']);jobs=[];skipped=[];input_bytes=0;total_frames=0
 # Walk parents of meta/data/videos, including damaged units without info.json.
 candidates=[]
 for root,dirs,files in os.walk(inp):
  dirs[:]=[x for x in dirs if not x.startswith('.')]
  if set(dirs)&{'meta','data','videos'}:candidates.append(Path(root));dirs[:]=[]
 selected_names={n for r in json.loads((P/'research/dimension_repairs.json').read_text())['rules'] for n in r['repositories']}
 not_selected=[]
 identities={};duplicates={}
 for source in candidates:
  try:cid=identify(source)['canonical_repo']
  except ValueError:continue
  identities.setdefault(cid,[]).append(str(source))
 duplicates={cid:paths for cid,paths in identities.items() if len(paths)>1}
 for s in sorted(candidates):
  if a.only_repairable and s.name not in selected_names:not_selected.append(str(s));continue
  try:
   repo=identify(s)
   assert repo['canonical_repo'] not in duplicates,'Duplicate HF/ModelScope or local copies of the same repository; select exactly one complete revision: '+str(duplicates.get(repo['canonical_repo']))
   info=json.loads((s/'meta/info.json').read_text());assert info['codebase_version']=='v2.1','Unverified source format'
   assert schema_key(info) in known or rule_for(s,info) is not None,'Numeric schema not covered by completed sample tests; skipped for review'
   assert (s/'data').is_dir() and (s/'videos').is_dir() and (s/'meta/tasks.jsonl').is_file(),'Missing required source paths'
   eps=[json.loads(x) for x in (s/'meta/episodes.jsonl').read_text().splitlines() if x.strip()];ids=[x['episode_index'] for x in eps]
   assert len(set(ids))==len(ids)==info['total_episodes'] and sum(x['length'] for x in eps)==info['total_frames'],'Metadata count mismatch'
   assert sorted(ids)==list(range(info['total_episodes'])),'Gapped episode ids require explicit review'
   for ep in eps:
    kw={'episode_index':ep['episode_index'],'episode_chunk':ep['episode_index']//info.get('chunks_size',1000)}
    paths=[s/info['data_path'].format(**kw)]+[s/info['video_path'].format(video_key=k,**kw) for k,v in info['features'].items() if v['dtype']=='video']
    assert all(p.is_file() and p.stat().st_size for p in paths),'Missing/empty referenced file'
   repair=validate_vectors(s,info)
   ranges=dict(info.get('splits',{}));covered=set()
   for key,span in ranges.items():
    assert '/' not in key and '..' not in key
    start,end=span.split(':');start=int(start or 0);end=int(end or len(ids));new=set(range(start,end));assert 0<=start<end<=len(ids) and not new&covered,'Invalid overlapping splits';covered|=new
   missing=sorted(set(ids)-covered)
   for ep in missing:ranges['unassigned_'+str(ep)]=f'{ep}:{ep+1}'
   fp=source_fingerprint(s);rel=str(s.relative_to(inp)) if s!=inp else s.name;identity=repo['canonical_name']+'__'+hashlib.sha256(rel.encode()).hexdigest()[:8]
   if info.get('discarded_episode_indices'):raise ValueError('Publisher discarded episodes require explicit selection review')
   for split in ranges:profile(s,info,split);jobs.append({'source':str(s),'id':identity+'__'+split,'split':split,'ranges':ranges,'source_fingerprint':fp,'metadata_repair':repair,'repository':repo})
   input_bytes+=sum(p.stat().st_size for p in input_files(s));total_frames+=info['total_frames']
  except (ValueError,KeyError,AssertionError,OSError,TypeError) as e:skipped.append({'source':str(s),'error':str(e)})
 expected={x['id'].split('/')[-1] for x in json.loads((P/'research/repos.json').read_text()) if x['id']!='RoboCOIN/pageAssets'}
 catalog_count=len(expected)
 if a.only_repairable:expected&=selected_names
 valid={j['repository']['canonical_name'] for j in jobs}
 plan={'selection':'repairable_only' if a.only_repairable else 'all','catalog_repos':catalog_count,'not_selected':not_selected,'missing_expected_repos':sorted(expected-valid),'expected_training_repos':len(expected),'jobs':jobs,'skipped':skipped,'candidate_repos':len(candidates),'valid_jobs':len(jobs),'input_bytes':input_bytes,'extra_disk_estimate_bytes':int(input_bytes*2)+4096*total_frames+2*1024**3,'duplicate_repositories':duplicates,'coverage_note':'All 809 catalog schemas supported; this plan reports actual local inputs only. Both platform names resolve to canonical IDs; revisions must not be mixed.'};save(work/'plan.json',plan)
 print(json.dumps({k:v for k,v in plan.items() if k not in ('jobs','missing_expected_repos')}),flush=True)
 if not jobs:sys.exit(2)
 if a.command=='plan':return
 assert shutil.disk_usage(out).free>=plan['extra_disk_estimate_bytes'],'Insufficient disk'
 results=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(job_run,j,out,work,a.video_workers,a.video_mb,a.decode_threads):j for j in jobs}
  for future in as_completed(futures):
   try:r=future.result()
   except Exception as e:r={'id':futures[future]['id'],'status':'failed','error':str(e),'traceback':traceback.format_exc()}
   results.append(r);save(work/'progress.json',{'results':results,'planned':len(jobs),'skipped':skipped});print(json.dumps(r),flush=True)
 sys.exit(1 if any(r['status']=='failed' for r in results) else 0)
if __name__=='__main__':main()
