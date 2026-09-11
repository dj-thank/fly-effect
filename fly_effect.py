"""Small, offline-first entry point for Fly Effect contributors."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from organism_core.config import DATA, HOME

def doctor(check_data=False):
    names=('numpy','brian2','mujoco','flygym','pandas','pyarrow','psutil')
    dependencies={name:importlib.util.find_spec(name) is not None for name in names}
    lock=json.loads((Path(__file__).parent/'organism_core/graph_lock.json').read_text(encoding='utf-8'))
    data={name:{'present':(DATA/'graph'/name).is_file()} for name in lock['files']}
    if check_data:
        for name,expected in lock['files'].items():
            if data[name]['present']:
                with (DATA/'graph'/name).open('rb') as f:data[name]['sha256_matches']=hashlib.file_digest(f,'sha256').hexdigest()==expected
    compatibility={'core_python_supported':sys.version_info>=(3,11),
                   'body_python_supported':(3,12)<=sys.version_info[:2]<(3,15),
                   'body_python_requirement':'>=3.12,<3.15 (pinned FlyGym)'}
    return {'runtime_compatibility':compatibility,'project':'Fly Effect','stage':'pre-alpha','dependencies':dependencies,'data_directory':str(DATA),'graph_files':data,'full_organism_validated':False,'note':'Doctor checks installation/data only. It does not certify a biological model or a full experiment.'}

def demo(out):
    import numpy as np
    import brian2 as b
    from organism_core.brain import Brain
    from organism_core.checkpoint import save,load
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    brain=Brain.synthetic();brain.cells.v[0]=-40*b.mV;brain.run(.001)
    identity={'kind':'synthetic','neurons':3,'biological_validation':False}
    receipt=save(out/'checkpoint.npz',brain.state(),identity)
    brain.run(.019);expected=brain.observation()
    brain.restore(load(receipt['path'],identity,receipt['sha256']));brain.run(.019)
    actual=brain.observation()
    match=all(np.array_equal(actual[k],expected[k]) for k in ('v_mV','g_mV','spike_i','spike_t'))
    np.savez_compressed(out/'observation.npz',**actual)
    result={'kind':'synthetic','neurons':3,'edges':2,'duration_s':.02,'spike_count':len(actual['spike_i']),'checkpoint_resume_equal':match,'biological_validation':False,'walking_claimed':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if not match:raise RuntimeError('Synthetic checkpoint regression')
    return result

def main():
    p=argparse.ArgumentParser(description='Fly Effect: reproducible neural/organism simulation research')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor');d.add_argument('--check-data',action='store_true')
    d=sub.add_parser('demo');d.add_argument('--out',type=Path,default=HOME/'work/synthetic-demo')
    d=sub.add_parser('mechanics-demo',help='Synthetic virtual-work positive/negative controls')
    d.add_argument('--out',type=Path,default=HOME/'work/mechanics-demo')
    d=sub.add_parser('calibrate-body',help='Matched artificial motor pulses; not autonomous walking')
    d.add_argument('--out',type=Path,default=HOME/'work/body-probe')
    d.add_argument('--duration',type=float,default=.02)
    d.add_argument('--wall-limit',type=float,default=120.)
    d.add_argument('--leg',choices=['lf','lm','lh','rf','rm','rh'],default='lf')
    d.add_argument('--joint-profile',choices=['generic','muscle_compliance','muscle_transfer'],default='muscle_compliance')
    d=sub.add_parser('audit-connectivity',help='Read-only locked graph census; no brain/body simulation')
    d.add_argument('--graph',type=Path,default=DATA/'graph')
    d.add_argument('--out',type=Path,required=True,help='New directory for result or failure receipt')
    d.add_argument('--chunk-rows',type=int,default=250000)
    d.add_argument('--wall-limit',type=float,default=120.)
    d.add_argument('--min-count',type=int,default=1)
    d.add_argument('--reachability',action='store_true',help='Explicitly enable bounded reverse graph traversal')
    d.add_argument('--max-passes',type=int,default=64)
    args=p.parse_args()
    if args.command=='doctor':result=doctor(args.check_data)
    elif args.command=='demo':result=demo(args.out)
    elif args.command=='audit-connectivity':
        from organism_core.connectivity_audit import audit_graph
        try:
            args.out.mkdir(parents=True,exist_ok=False)
        except OSError as error:
            p.error(str(error))
        try:
            result=audit_graph(args.graph,chunk_rows=args.chunk_rows,wall_limit=args.wall_limit,
                               min_count=args.min_count,reachability=args.reachability,max_passes=args.max_passes)
        except (OSError,ValueError,TypeError) as error:
            result={'schema':1,'status':'incomplete' if isinstance(error,TimeoutError) else 'unavailable_or_invalid',
                    'error':str(error),'measured_census':None,'biological_validation':False,'walking_claimed':False}
            (args.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            p.error(str(error))
        (args.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    elif args.command=='mechanics-demo':
        from organism_core.calibration import run_mechanics_demo
        result=run_mechanics_demo(args.out)
    else:
        if not (3,12)<=sys.version_info[:2]<(3,15):
            p.error('The pinned FlyGym body backend requires Python >=3.12,<3.15; core/neural examples also support 3.11.')
        from organism_core.calibration import run_body_probe
        result=run_body_probe(args.out,duration=args.duration,joint_profile=args.joint_profile,
                              leg=args.leg,wall_limit=args.wall_limit)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
