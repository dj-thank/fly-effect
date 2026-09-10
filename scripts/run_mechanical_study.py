"""One explicit bounded run: verify source -> register 90 tendons -> audit/probe.

Uses a NEW workspace; never overwrites existing calibration or neural data.
The two source-generation scripts are repository tools, not wheel entrypoints.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--duration',type=float,default=.25)
    args=p.parse_args()
    if not 0.15 <= args.duration <= .5 or abs(args.duration-round(args.duration/.0001)*.0001)>1e-12:
        p.error('Duration must be 0.15--0.5 seconds in integer 100-us ticks')
    workspace=args.workspace.resolve(); workspace.mkdir(parents=True,exist_ok=False)
    (workspace/'runs').mkdir()
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,FLY_EFFECT_HOME=str(workspace))
    try:
        for script in ('audit_source_tendons.py','transfer_six_leg_tendons.py'):
            with (workspace/(script+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(root/script)],cwd=root,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        # Run in a fresh interpreter so config never retains a previous workspace.
        command='from organism_core.tendon_study import run_study; import sys; run_study(sys.argv[1],duration=float(sys.argv[2]))'
        subprocess.run([sys.executable,'-c',command,str(workspace/'study'),str(args.duration)],
                       cwd=root,env=env,check=True,timeout=300)
        command='from organism_core.pose_probe import run_pose_probe; import sys; run_pose_probe(sys.argv[1],sys.argv[2])'
        subprocess.run([sys.executable,'-c',command,str(workspace/'pose-study'),str(workspace/'study')],
                       cwd=root,env=env,check=True,timeout=180)
    except Exception as exc:
        (workspace/'pipeline-failure.json').write_text(json.dumps(
            {'status':'failed','error_type':type(exc).__name__,'error':str(exc),
             'biological_validation':False,'walking_claimed':False},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
