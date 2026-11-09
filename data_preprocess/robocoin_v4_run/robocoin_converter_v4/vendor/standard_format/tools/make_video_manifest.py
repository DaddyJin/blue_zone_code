"""One clip per episode -> explicit video-only adapter manifest."""
import argparse,json,hashlib
from pathlib import Path
from media import probe
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('videos',nargs='+');p.add_argument('--output',required=True);p.add_argument('--dataset-id',required=True);p.add_argument('--policy',choices=['native','normalize'],required=True);p.add_argument('--fps',type=float,required=True);p.add_argument('--size',nargs=2,type=int,metavar=('WIDTH','HEIGHT'));a=p.parse_args()
    if a.policy=='normalize' and not a.size:p.error('normalize requires --size WIDTH HEIGHT')
    episodes=[]
    for i,value in enumerate(a.videos):
        v=Path(value).resolve();h=probe(v)
        episodes.append({'source_episode_id':v.stem,'videos':{'main':str(v)},'fps':h['fps'],'task':'','split_group_uid':f'{a.dataset_id}:source-video:'+hashlib.sha256(str(v).encode()).hexdigest()})
    c={'dataset_id':a.dataset_id,'profile_id':a.dataset_id+':video-only:v2','data_kind':'video_only','robot_type':'none','fps':a.fps,'media':{'policy':a.policy},'state_fields':[],'action':{'mode':'none'},'time':{'kind':'unknown'},'source':{'id':a.dataset_id,'revision':'unknown; replace with release or file checksums','notes':'Video PTS is presentation time; no physical capture clock supplied'},'episodes':episodes}
    if a.policy=='normalize':c['media'].update(fps=a.fps,size=a.size)
    Path(a.output).write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');print(a.output)
