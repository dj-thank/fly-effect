"""One local entry point; unfinished functions fail explicitly."""
from pathlib import Path
import argparse
import contextlib
from datetime import datetime,timezone
import json
import os
import time
import uuid
import sys
import subprocess
import zipfile

from organism_core.config import HOME as ROOT, SOURCE_ROOT
os.environ['PYTHONUTF8']='1'
os.environ.setdefault('FLYGYM_ASSET_CACHE_DIR',str(ROOT/'cache/flygym-assets'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
temp=ROOT/'cache/tmp'
temp.mkdir(parents=True,exist_ok=True)
os.environ['TEMP']=str(temp)
os.environ['TMP']=str(temp)

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

@contextlib.contextmanager
def lease():
    import psutil
    path=ROOT/'run.lock'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        try:alive=abs(psutil.Process(old['pid']).create_time()-old['created'])<.01
        except psutil.NoSuchProcess:alive=False
        if alive:raise RuntimeError('A task-owned simulation is already running')
        path.rename(ROOT/('stale-run-'+uuid.uuid4().hex+'.json'))
    current=psutil.Process()
    with path.open('x',encoding='utf-8') as f:
        json.dump({'pid':current.pid,'created':current.create_time(),'owner':'fly-effect','cwd':str(ROOT)},f)
    try:yield
    finally:path.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('run','resume'):
        p=sub.add_parser(name);p.add_argument('--duration',type=float,default=.01);p.add_argument('--seed',type=int,default=0)
        p.add_argument('--out',type=Path);p.add_argument('--checkpoint-at',type=float)
        p.add_argument('--wall-limit',type=float,default=600)
        p.add_argument('--sensory-mode',choices=['six_leg','six_leg_load','legacy'],default='six_leg')
        p.add_argument('--joint-profile',choices=['generic','muscle_compliance','muscle_transfer'],default='generic')
        p.add_argument('--muscle-model',choices=['antagonist','tendon_candidate'],default='antagonist')
        p.add_argument('--load-cut',action='store_true')
        p.add_argument('--input-cut',action='store_true')
        p.add_argument('--perturb-leg',choices=['lf','lm','lh','rf','rm','rh'])
        p.add_argument('--perturb-at',type=float,default=.05)
        if name=='resume':p.add_argument('--receipt',type=Path,required=True)
    sub.add_parser('status')
    p=sub.add_parser('evaluate');p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    for name in ('view','record'):sub.add_parser(name)
    args=parser.parse_args()
    if args.action=='status':
        print(json.dumps({'stage':'pre-alpha','full_organism_validated':False,'data_directory':str(__import__('organism_core.config',fromlist=['DATA']).DATA)},indent=2));return
    if args.action in ('view','record'):raise SystemExit('Not implemented yet; no visual evidence claimed')
    if args.action=='evaluate':
        left=json.loads(args.left.read_text(encoding='utf-8'));right=json.loads(args.right.read_text(encoding='utf-8'))
        keys=('identity','ticks','input_count','observation_hashes','rng')
        checks={key:left[key]==right[key] for key in keys}
        print(json.dumps({'resume_equivalent':all(checks.values()),'checks':checks,'walking_passed':False},indent=2))
        if not all(checks.values()):raise SystemExit(1)
        return
    if not 0<args.duration<=3 or not 0<args.wall_limit<=1800:raise ValueError('Bounded duration/wall limit required')
    out=args.out.resolve() if args.out else ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not out.is_relative_to(ROOT):raise ValueError('Run output must stay inside the project')
    out.mkdir(parents=True,exist_ok=False)
    with lease():
        began=time.monotonic()
        from organism_core.engine import Engine
        prior=json.loads(args.receipt.read_text(encoding='utf-8')) if args.action=='resume' else None
        settings={'sensory_mode':args.sensory_mode,'input_enabled':not args.input_cut,'perturbation':None,'joint_profile':args.joint_profile,'muscle_model':args.muscle_model,'load_enabled':not args.load_cut}
        if args.perturb_leg:
            if not 0<=args.perturb_at<args.duration:raise ValueError('Perturbation time outside trial')
            settings['perturbation']={'leg':args.perturb_leg,'tick':round(args.perturb_at/.0001),'velocity_kick_rad_s':5.}
        if prior:settings=prior['settings']
        engine=Engine(prior['seed'] if prior else args.seed,settings)
        if prior:engine.restore(Path(prior['checkpoint']['path']),prior['checkpoint']['sha256'])
        dump(out/'identity.json',engine.identity)
        dump(out/'assets.json',engine.body.asset_files)
        (out/'body.xml').write_text(engine.body.xml,encoding='utf-8',newline='\n')
        with zipfile.ZipFile(out/'runtime-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in [SOURCE_ROOT/'organism.py',*sorted((SOURCE_ROOT/'organism_core').glob('*.py'))]:
                archive.write(path,path.relative_to(SOURCE_ROOT).as_posix())
            for source in engine.body.model_source_hashes:
                path=Path(source)
                if path.is_relative_to(ROOT):archive.write(path,path.relative_to(ROOT).as_posix())
        remaining=args.duration
        first=True
        while remaining>1e-10:
            chunk=min(remaining,args.checkpoint_at if first and args.checkpoint_at else .01)
            engine.run(chunk);remaining-=chunk;first=False
            if args.checkpoint_at and not (out/'checkpoint.json').exists():
                checkpoint=engine.snapshot(out/'checkpoint.npz')
                dump(out/'checkpoint.json',{'checkpoint':checkpoint,'seed':engine.seed,'tick':engine.tick,'settings':engine.settings})
            dump(out/'progress.json',{'tick':engine.tick,'wall_seconds':time.monotonic()-began})
            if time.monotonic()-began>args.wall_limit:
                checkpoint=engine.snapshot(out/'timeout-checkpoint.npz')
                dump(out/'timeout.json',{'checkpoint':checkpoint,'seed':engine.seed,'settings':engine.settings,'reason':'wall budget'})
                raise SystemExit('Wall limit reached; checkpoint saved')
        report=engine.report();report['wall_seconds']=time.monotonic()-began
        import numpy as np
        observation=engine.observation()
        np.savez_compressed(out/'observation.npz',**{k:v for k,v in observation.items() if isinstance(v,np.ndarray)})
        dump(out/'result.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('motor_accounting','observation_hashes','rng','identity')},indent=2))

if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        env=dict(os.environ,PYTHONHASHSEED='0')
        raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env=env))
    main()
