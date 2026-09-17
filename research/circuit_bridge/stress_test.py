"""Frozen-surrogate challenges outside the original two-channel stimulus family.

This is a prespecified follow-on model experiment, not new biological validation.
Never refit the POD basis or choose regions/ranks on these challenge responses.
Sources for neuronal IDs are the upstream source annotations. Raw bytes are
verified by import_graph(); the original baseline evidence supplies the basis.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from .connectome import SparseCircuit, RateConfig, simulate, comparison, array_hash
from .malecns import import_graph, stimuli, write_json, SOURCE_SHA, SOURCES, file_hash

GROUPS = {
    'DNa02': (10360,523769),
    'DNg13': (11074,512006),
    'MDN': (10763,11288,11332,12348),
}


def resolve_ids(ids, wanted):
    index = {int(value):i for i,value in enumerate(ids)}
    if len(set(wanted)) != len(wanted) or not all(value in index for value in wanted):
        raise ValueError('Missing or duplicate source body IDs')
    return np.array([index[value] for value in wanted], dtype=np.int64)


def challenge_drive(name, steps=160):
    """Fixed novel stimulus protocols: no optimizer or held-out-response feedback."""
    if name == 'fast_alternating':
        t=np.arange(steps)
        return np.column_stack((.9*(t%8<4),.9*(t%8>=4)))
    if name == 'long_silence_and_burst':
        out=np.zeros((steps,2))
        out[10:30,0]=1.1;out[70:110,1]=.9;out[135:,:]=.8
        return out
    if name in GROUPS:
        rng=np.random.default_rng(2027)
        out=np.repeat(rng.uniform(0,1.1,(20,len(GROUPS[name]))),8,axis=0)[:steps]
        out[:10]=0
        return out
    raise ValueError('Unknown prespecified challenge')


def run_challenges(directory, baseline_directory, outdir):
    d,base,outdir=map(Path,(directory,baseline_directory,outdir))
    # Never accept changed source bytes on follow-on experiments.
    for key,name in SOURCES.items():
        if file_hash(d/name)!=SOURCE_SHA[key]:raise ValueError('Source byte mismatch')
    ids,edges,signs,motor,original_stim,graph=import_graph(d)
    baseline=json.loads((base/'result.json').read_text())
    exp=baseline['experiment']
    with np.load(base/'traces/training.npz',allow_pickle=False) as record:
        basis=record['basis'].copy();region_ids=record['region_ids'].copy()
    if array_hash(basis)!=exp['basis_sha256']:raise ValueError('Changed frozen basis')
    if region_ids.tolist()!=exp['region_body_ids']:raise ValueError('Changed frozen region')
    region=resolve_ids(ids,region_ids.tolist())
    config=RateConfig(**exp['config'])
    circuit=SparseCircuit(ids,edges['pre'],edges['post'],edges['count'],signs,config)
    if circuit.fingerprint!=exp['graph_sha256']:raise ValueError('Changed baseline graph')
    random=np.linalg.qr(np.random.default_rng(991).normal(size=basis.shape))[0]
    from .malecns import table_frame
    annotations=table_frame(d/SOURCES['annotations']).set_index('bodyId')
    outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in ('fast_alternating','long_silence_and_burst',*GROUPS):
        print('FROZEN_CHALLENGE',name,flush=True)
        stim=resolve_ids(ids,GROUPS[name]) if name in GROUPS else original_stim
        if name in GROUPS and not annotations.loc[ids[stim],'type'].eq(name).all():
            raise ValueError('Stimulus source annotations disagree with challenge identity')
        drive=challenge_drive(name)
        ref=simulate(circuit,region,motor,stim,drive)
        les=simulate(circuit,region,motor,stim,drive,'lesion')
        np.savez_compressed(outdir/f'{name}-intact.npz',motor=ref['motor'],region=ref['region'],drive=drive)
        for mode,u in (('lesion',None),('pod',basis),('random_basis',random)):
            result=les if mode=='lesion' else simulate(circuit,region,motor,stim,drive,mode,u)
            score=comparison(ref['motor'],les['motor'],result['motor'])
            np.savez_compressed(outdir/f'{name}-{mode}.npz',motor=result['motor'],region=result['region'],drive=drive)
            norm=score['reference_norm']
            rows.append({'challenge':name,'mode':mode,'stimulus_body_ids':ids[stim].tolist(),
                'stimulus_sha256':result['stimulus_hash'],'motor_trace_sha256':result['motor_hash'],
                **score,'lesion_relative_reference_error':score['lesion_error']/norm if norm>1e-12 else None})
    report={'schema':'circuit-bridge-frozen-challenges/v1','graph':graph,
        'basis_sha256':array_hash(basis),'baseline_selected_rank':exp['selected_rank'],
        'region_body_ids':region_ids.tolist(),'retrained':False,
        'config':exp['config'],'challenge_seed':2027,'rows':rows,
        'claims':{'biological_validation':False,'body_simulated':False,'walking_demonstrated':False},
        'interpretation':'Follow-on challenge of one frozen surrogate, not an unbiased confirmatory study or population estimate.'}
    write_json(outdir/'challenges.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run_challenges(a.data,a.baseline,a.out)


if __name__=='__main__':main()
