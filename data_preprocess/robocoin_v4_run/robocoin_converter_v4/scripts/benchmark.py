"""Benchmark an already-extracted representative subset; never use full corpus for tuning."""
from pathlib import Path
import subprocess,time,json,os,sys,argparse,resource
P=Path(__file__).resolve().parents[1]
def rss_tree(pid):
 table={}
 for f in Path('/proc').glob('[0-9]*/status'):
  try:
   d={line.split(':')[0]:line.split(':',1)[1].strip() for line in f.read_text().splitlines() if ':' in line};table[int(f.parent.name)]=(int(d['PPid']),int(d.get('VmRSS','0 kB').split()[0]))
  except (OSError,ValueError,KeyError):pass
 seen={pid}
 while True:
  new=seen|{p for p,(parent,rss) in table.items() if parent in seen}
  if new==seen:break
  seen=new
 return sum(table.get(p,(0,0))[1] for p in seen)*1024

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--configs',default='4x4,8x4,16x4,32x4');a=p.parse_args()
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:128]);env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');results=[]
 for label in a.configs.split(','):
  parts=list(map(int,label.split('x')));w,v=parts[:2];d=parts[2] if len(parts)>2 else 1;out=a.output/label;work=a.work/label
  if out.exists():raise FileExistsError('Benchmark requires fresh output: '+str(out))
  work.mkdir(parents=True);start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0
  with (work/'run.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,str(P/'scripts/convert_repos.py'),'convert','--input',str(a.input),'--output',str(out),'--work',str(work),'--workers',str(w),'--video-workers',str(v),'--decode-threads',str(d)],env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:peak=max(peak,rss_tree(proc.pid));time.sleep(.25)
  elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN);plan=json.loads((work/'plan.json').read_text());frames=sum(json.loads((out/j['id']/'SUCCESS.json').read_text())['frames'] for j in plan['jobs'] if (out/j['id']/'SUCCESS.json').is_file())
  result={'workers':w,'video_workers':v,'decode_threads':d,'seconds':elapsed,'returncode':proc.returncode,'sampled_sum_rss_gib':peak/2**30,'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'rows':frames,'rows_per_second':frames/elapsed,'source_mib':plan['input_bytes']/2**20,'source_mib_per_second':plan['input_bytes']/2**20/elapsed,'output':str(out)};results.append(result);print(json.dumps(result),flush=True)
  a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps({'scope':'Warm-cache representative subset; whole pipeline includes full video decode and SHA; RSS is sampled process-tree sum, not unique physical memory; not a full-corpus throughput guarantee','results':results},indent=2))
 sys.exit(1 if any(x['returncode'] for x in results) else 0)
if __name__=='__main__':main()
