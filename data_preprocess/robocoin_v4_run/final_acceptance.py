"""Fail closed for full-catalog acceptance; no data is modified."""
import argparse,json,sys,collections
from pathlib import Path

def evaluate(work, output):
    errors=[];summary={}
    def check(ok,message):
        if not ok:errors.append(message)
    def read(name):return json.loads((work/name).read_text())
    try:
        plan=read('plan.json');progress=read('progress.json');verify=read('verification.json');load=read('official_load.json')
        check(plan.get('selection')=='all','Plan is not full selection')
        check(plan.get('expected_training_repos')==809,'Expected catalog is not 809 repositories')
        check(not plan.get('skipped'),'Plan contains skipped inputs')
        check(not plan.get('missing_expected_repos'),'Plan has missing repositories')
        check(not plan.get('duplicate_repositories'),'Duplicate repository versions detected')
        jobs=plan['jobs'];ids=[j['id'] for j in jobs];expected=set(ids)
        check(bool(ids) and len(ids)==len(expected),'Empty or duplicate plan jobs')
        repos={j['repository']['canonical_repo'] for j in jobs}
        check(len(repos)==809,'Unique planned repositories != 809')
        check(plan.get('valid_jobs')==len(jobs),'Plan job count mismatch')
        check(progress.get('planned')==len(jobs),'Progress planned count mismatch')
        check(not progress.get('skipped'),'Progress contains skipped inputs')
        for label,rows,key in [('progress',progress['results'],'id'),('verification',verify['units'],'id'),('loader',load['units'],'repo')]:
            found=[r[key] for r in rows]
            check(len(found)==len(set(found)) and set(found)==expected,f'{label} job set mismatch')
        check(all(r['status'] in ('complete','skipped_verified') for r in progress['results']),'Incomplete or failed conversion jobs')
        check(verify.get('passed') is True and verify.get('corpus_complete') is True,'Verification is not fully complete')
        check(not verify.get('skipped') and not verify.get('missing_expected_repos'),'Verification reports omissions')
        check(all(r.get('passed') is True for r in verify['units']),'Failed verification unit')
        check(load.get('passed') is True and load.get('lerobot_version')=='0.5.0','LeRobot 0.5.0 report failed')
        check(all(r.get('passed') is True for r in load['units']),'Failed loader unit')
        actual={p.name for p in output.iterdir() if p.is_dir() and not p.name.startswith('.')}
        check(actual==expected,'Output directory set differs from current full plan')
        total_frames=0;total_episodes=0;origins=collections.Counter()
        verified={r['id']:r for r in verify['units']}
        for jid in ids:
            root=output/jid
            check((root/'SUCCESS.json').is_file() and (root/'conversion_receipt.json').is_file(),f'{jid}: missing success or receipt')
            info=json.loads((root/'meta/info.json').read_text())
            check(info.get('codebase_version')=='v3.0',f'{jid}: not LeRobot v3')
            check(verified.get(jid,{}).get('rows')==info['total_frames'],f'{jid}: frame count mismatch')
            episodes=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
            check(len(episodes)==info['total_episodes'],f'{jid}: episode count mismatch')
            check(sum(e['length'] for e in episodes)==info['total_frames'],f'{jid}: episode lengths mismatch')
            for e in episodes:origins[e['action_semantics']['mode']]+=1
            total_frames+=info['total_frames'];total_episodes+=info['total_episodes']
        summary={'repositories':len(repos),'output_units':len(ids),'episodes':total_episodes,'frames':total_frames,'action_episode_counts':dict(origins),'hidden_staging_directories':[p.name for p in output.iterdir() if p.is_dir() and p.name.startswith('.')]}
    except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
    return {'passed':not errors,'scope':'809-repository full catalog; requires freshly run verify and load on unchanged output','errors':errors,**summary}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=evaluate(a.work,a.output);dest=a.work/'acceptance_v4.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2));tmp.replace(dest)
    print(json.dumps(r,ensure_ascii=False,indent=2));sys.exit(0 if r['passed'] else 1)
