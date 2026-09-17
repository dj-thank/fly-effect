"""Frozen-rank, held-out and out-of-distribution tests on the full locked graph.

Not a biological experiment. Each condition recomputes live recurrent neural
feedback. A low-rank surrogate retains the internal operator and all output
ports; reduced latent dimension is NOT a whole-brain speedup claim.
"""
from dataclasses import asdict
import numpy as np
from scipy import sparse
from .connectome import RateConfig, SparseCircuit, array_hash, comparison, pod_basis, simulate, internal_shuffle


def stimuli(seed, steps=120, amplitude=.7):
    rng = np.random.default_rng(seed)
    levels = rng.uniform(0, amplitude, size=((steps+14)//15, 2))
    drive = np.repeat(levels, 15, axis=0)[:steps]
    drive[:10] = 0
    drive[steps//2:steps//2+10, seed % 2] = 0
    return drive


def stress_stimuli(seed, region_size, port_amplitude=.06):
    """Independent region-port inputs challenge a basis learned from two DNs."""
    rng = np.random.default_rng(seed + 10000)
    port = np.repeat(rng.uniform(-port_amplitude, port_amplitude,
                                 size=(8, region_size)),15,axis=0)
    port[:10] = 0
    return np.column_stack((stimuli(seed, amplitude=1.1),port))


def choose_rank(validation, threshold=.95):
    eligible = [x['rank'] for x in validation if x['recovery_fraction'] is not None
                and x['recovery_fraction'] >= threshold]
    rank = min(eligible) if eligible else min(validation,key=lambda x:x['candidate_error'])['rank']
    return rank, bool(eligible)


def run(ids, edges, signs, motor, stim, gain=.8, traces=None):
    config = RateConfig(gain=gain)
    circuit = SparseCircuit(ids, edges['pre'], edges['post'], edges['count'], signs, config)
    unsigned = sparse.csr_matrix((edges['count'].astype(float),(edges['post'],edges['pre'])),
                                  shape=(len(ids),len(ids)))
    incoming = np.asarray(unsigned[:,stim].sum(axis=1)).ravel()
    outgoing = np.asarray(unsigned[motor].sum(axis=0)).ravel()
    score = incoming*outgoing
    score[np.concatenate((motor,stim))] = 0
    order = np.lexsort((ids,-score))
    region = np.sort(order[score[order] > 0][:64])
    if len(region) != 64:
        raise ValueError('Prespecified 64-node bridge population not available')
    del unsigned
    print(f'Full-graph bridge gain={gain}: 64 anatomical intermediate neurons',flush=True)
    train_seeds, val_seed, test_seeds = (0,1),11,(101,102,103)
    training = np.vstack([simulate(circuit,region,motor,stim,stimuli(s))['region'] for s in train_seeds])
    bases = {rank:pod_basis(training,rank) for rank in (2,4,8,16)}
    validation = []
    drive = stimuli(val_seed)
    reference = simulate(circuit,region,motor,stim,drive)['motor']
    lesion = simulate(circuit,region,motor,stim,drive,'lesion')['motor']
    for rank,basis in bases.items():
        candidate = simulate(circuit,region,motor,stim,drive,'pod',basis)['motor']
        validation.append({'seed':val_seed,'rank':rank,**comparison(reference,lesion,candidate)})
    chosen, passed = choose_rank(validation)
    basis = bases[chosen]
    random = np.linalg.qr(np.random.default_rng(991).normal(size=(64,chosen)))[0]
    trials = []
    if traces is not None:
        traces['training_region'] = training
        traces['basis'] = basis
        traces['motor_body_ids'] = ids[motor]
        traces['region_body_ids'] = ids[region]

    def evaluate(cohort, seed, indices, drive, conditions):
        reference = simulate(circuit,region,motor,indices,drive)
        lesion = simulate(circuit,region,motor,indices,drive,'lesion')
        prefix = f'g{gain:g}_{cohort}_{seed}'
        if traces is not None:
            traces[prefix+'_reference'] = reference['motor']
            traces[prefix+'_drive'] = drive
        for label,mode,basis_arg in conditions:
            candidate = lesion if mode=='lesion' else simulate(circuit,region,motor,indices,drive,mode,basis_arg,shuffle_seed=seed)
            if label=='native' and not np.allclose(reference['motor'],candidate['motor'],atol=1e-12,rtol=1e-10):
                raise AssertionError('Native replacement changed full-graph response')
            per_step_error = np.max(np.abs(reference['motor']-candidate['motor']),axis=1)
            bound_violation = float(np.max(per_step_error-candidate['error_bound_inf']))
            if bound_violation > 1e-12:
                raise AssertionError('Analytic defect bound failed numerical motor check')
            trial = {'cohort':cohort,'seed':seed,'mode':label,
                **comparison(reference['motor'],lesion['motor'],candidate['motor']),
                'maximum_motor_error':float(per_step_error.max()),
                'certificate':candidate['certificate'],
                'bound_check':{'passed':True,'max_violation':bound_violation,'tolerance':1e-12},
                'stimulus_sha256':candidate['stimulus_hash'],
                'motor_trace_sha256':candidate['motor_hash']}
            if traces is not None:
                key = prefix+'_'+label
                traces[key] = candidate['motor']
                traces[key+'_bound'] = candidate['error_bound_inf']
                traces[key+'_defect'] = candidate['defect_inf']
                trial['trace_key'] = key
            trials.append(trial)
        print(f'Completed {cohort} seed {seed}, gain {gain}',flush=True)

    for seed in test_seeds:
        evaluate('heldout_dn',seed,stim,stimuli(seed,amplitude=1.1),
                 [('lesion','lesion',None),('native','native',None),('pod','pod',basis),
                  ('shuffled','shuffled',None),('random_basis','random_basis',random)])

    # Robust basis is trained and selected without access to stress-test traces.
    extended = np.concatenate((stim,region))
    stress_training = np.vstack([training]+[simulate(circuit,region,motor,extended,
        stress_stimuli(seed,len(region)))['region'] for seed in (21,22)])
    robust_bases = {rank:pod_basis(stress_training,rank) for rank in (2,4,8,16,32,64)}
    drive = stress_stimuli(31,len(region))
    reference = simulate(circuit,region,motor,extended,drive)['motor']
    lesion = simulate(circuit,region,motor,extended,drive,'lesion')['motor']
    stress_validation = []
    for rank,u in robust_bases.items():
        candidate = simulate(circuit,region,motor,extended,drive,'pod',u)['motor']
        stress_validation.append({'seed':31,'rank':rank,**comparison(reference,lesion,candidate)})
    robust_rank, robust_passed = choose_rank(stress_validation)
    robust_basis = robust_bases[robust_rank]
    if traces is not None:
        traces['stress_training_region'] = stress_training
        traces['stress_basis'] = robust_basis
    robust_random = np.linalg.qr(np.random.default_rng(1991).normal(size=(64,robust_rank)))[0]
    for seed in (201,202,203):
        evaluate('heldout_port_stress',seed,extended,stress_stimuli(seed,len(region),port_amplitude=.10),
                 [('lesion','lesion',None),('native','native',None),('frozen_pod','pod',basis),
                  ('robust_pod','pod',robust_basis),('robust_random_basis','random_basis',robust_random)])
    internal = circuit.w[region][:,region].tocsr(); internal.eliminate_zeros()
    changed = [(internal != internal_shuffle(internal,seed)).nnz for seed in test_seeds]
    return {'model':'contractive dimensionless signed rate model; not Brian2 LIF',
        'config':asdict(config),'graph_sha256':circuit.fingerprint,
        'region_selection':'top 64 anatomical two-hop DNge104->candidate->motor count products; exclude DN and motor; tie-break source ID',
        'region_size':64,'region_body_ids':ids[region].tolist(),
        'internal_functional_edges':int(internal.nnz),
        'shuffle_changed_matrix_entries':list(map(int,changed)),
        'train_seeds':list(train_seeds),'validation_seeds':[val_seed],'test_seeds':list(test_seeds),
        'training_trace_sha256':array_hash(training),'basis_sha256':array_hash(basis),
        'selected_rank':chosen,'latent_state_reduction_factor':64/chosen,
        'selection_passed_validation':passed,'validation':validation,
        'stress':{'train_seeds':[21,22],'validation_seeds':[31],'test_seeds':[201,202,203],
            'train_port_amplitude':.06,'test_port_amplitude':.10,
            'selected_rank':robust_rank,'latent_state_reduction_factor':64/robust_rank,
            'selection_passed_validation':robust_passed,'validation':stress_validation,
            'training_trace_sha256':array_hash(stress_training),'basis_sha256':array_hash(robust_basis)},
        'trials':trials,
        'limitations':[
            'Artificial direct neural drive, not natural sensation or behavior.',
            'One specimen. Seeds are stimulus patterns, not independent animals.',
            'Contractive rate dynamics intentionally cannot reproduce autonomous sustained rhythms.',
            'Unknown/unmodelled transmitters have zero functional weight. Anatomy is retained.',
            'POD retains the internal anatomical operator and reconstructs every output port.',
            'Latent dimension reduction is not measured speedup or deletion of anatomical neurons.',
            'Rewiring preserves directed degrees/source weight multisets, not incoming weighted strength.',
            'The online bound is for the declared mathematical model, not biology or rigorous floating-point intervals.',
            'Small finite seed sets do not establish generalization to arbitrary stimuli.']}


def main():
    import argparse, json, os, platform, time
    from pathlib import Path
    from .malecns import acquire, import_graph, SOURCES, SOURCE_SHA, file_hash, write_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reuse',action='store_true')
    p.add_argument('--gains',nargs='+',type=float,default=[.5,.8,.95])
    args=p.parse_args()
    if len(set(args.gains))!=len(args.gains):
        raise ValueError('Duplicate gain hypotheses')
    for gain in args.gains: RateConfig(gain=gain)
    started=time.monotonic()
    source=json.loads((args.data/'source-manifest.json').read_text()) if args.reuse else acquire(args.data)
    if set(source.get('files',{}))!=set(SOURCES):
        raise ValueError('Incomplete source manifest')
    for key,item in source['files'].items():
        if item['sha256']!=SOURCE_SHA[key] or file_hash(args.data/SOURCES[key])!=SOURCE_SHA[key]:
            raise ValueError('Pinned source bytes/manifest mismatch')
    ids,edges,signs,motor,stim,graph=import_graph(args.data)
    graph.pop('raw_source_rows_preserved',None)
    graph['source_row_indices_preserved']=True
    write_json(args.data/'graph-audit.json',graph)
    result={'schema':'circuit-bridge-certified/v1','status':'running','sources':source,'graph':graph,
        'planned_gains':args.gains,'experiments':{},
        'claims':{'real_connectome_loaded':True,'full_annotated_graph_retained':True,
            'biological_validation':False,'living_tissue_connected':False,'body_simulated':False,
            'walking_demonstrated':False,'upstream_lif_executed':False,
            'rigorous_floating_point_enclosure':False,'whole_brain_speedup_demonstrated':False},
        'source_commit':os.environ.get('GITHUB_SHA'),'actions_run_id':os.environ.get('GITHUB_RUN_ID'),
        'software':{'python':platform.python_version(),'numpy':np.__version__}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.out,result)
    try:
        for gain in args.gains:
            traces={} if gain==.8 else None
            experiment=run(ids,edges,signs,motor,stim,gain,traces)
            if traces is not None:
                archive=args.out.parent/'traces-gain-0.8.npz'
                np.savez_compressed(archive,**traces)
                experiment['trace_archive']={'file':archive.name,'sha256':file_hash(archive)}
            result['experiments'][format(gain,'g')]=experiment
            result['elapsed_seconds']=time.monotonic()-started
            write_json(args.out,result)
    except Exception as exc:
        result['status']='failed'
        result['failure']={'type':type(exc).__name__,'message':str(exc)}
        write_json(args.out,result)
        raise
    result['status']='complete'
    result['elapsed_seconds']=time.monotonic()-started
    write_json(args.out,result)
    print('CERTIFIED_BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':
    main()
