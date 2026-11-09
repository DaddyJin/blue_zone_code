"""Separate environment runner: real LeRobot 0.5.0 plus included compatibility subclass."""
from pathlib import Path
import subprocess,json,os,time,sys,argparse
P=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--python',required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1');rows=[]
 for unit in sorted(a.output.iterdir()):
  if not (unit/'SUCCESS.json').exists():continue
  start=time.monotonic()
  try:
   r=subprocess.run([a.python,str(P/'scripts/loader_probe.py'),str(unit)],env=env,capture_output=True,text=True,timeout=300)
   record={'repo':unit.name,'passed':r.returncode==0,'seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr}
  except Exception as e:record={'repo':unit.name,'passed':False,'error':str(e)}
  rows.append(record);print(unit.name,record['passed'],flush=True)
 report={'passed':bool(rows) and all(x['passed'] for x in rows),'lerobot_version':'0.5.0','compatibility':'subclass for missing subtask sentinel and original singleton vector types','units':rows};a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
if __name__=='__main__':main()
