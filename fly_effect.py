"""Small, offline-first entry point for Fly Effect contributors."""
import argparse
import hashlib
import importlib.util
import json
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
    return {'project':'Fly Effect','stage':'pre-alpha','dependencies':dependencies,'data_directory':str(DATA),'graph_files':data,'full_organism_validated':False,'note':'Doctor checks installation/data only. It does not certify a biological model or a full experiment.'}

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
    args=p.parse_args()
    result=doctor(args.check_data) if args.command=='doctor' else demo(args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
