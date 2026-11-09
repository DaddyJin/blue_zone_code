"""Inventory video headers without re-encoding or modifying sources."""
import argparse,json
from pathlib import Path
from collections import Counter
from media import probe

def inventory(paths):
    rows=[]
    for value in paths:
        p=Path(value).resolve();files=sorted(p.rglob('*.mp4')) if p.is_dir() else [p]
        for f in files:
            try:
                h=probe(f);rows.append({'path':str(f),'width':h['width'],'height':h['height'],'fps':h['fps'],'codec':h['codec_name'],'frames':h['frames'],'duration':h['duration'],'header_ok':True})
            except Exception as exc:rows.append({'path':str(f),'header_ok':False,'error':str(exc)})
    profiles=Counter((r['width'],r['height'],r['fps'],r['codec']) for r in rows if r['header_ok'])
    return {'files':rows,'profiles':[dict(width=w,height=h,fps=f,codec=c,files=n) for (w,h,f,c),n in profiles.items()],'note':'Headers only. Group by delivery unit and camera before deciding native vs normalize; full decode is a later required check.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--output');a=p.parse_args();r=inventory(a.paths);text=json.dumps(r,ensure_ascii=False,indent=2)
    if a.output:Path(a.output).write_text(text+'\n')
    print(text)
