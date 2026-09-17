"""Independently audit saved numerical evidence; never treat green CI as a result.

Usage: python -m research.circuit_bridge.verify_evidence --evidence DIR --out audit.json
The directory must contain result.json, execution.json, source-commit.txt and
traces/*.npz. No original graph or network download is needed to recheck metrics.
This verifies recorded artifacts, NOT the simulation equations or biology.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np

MODES = ('lesion', 'native', 'pod', 'shuffled', 'random_basis')
LIMITED_CLAIMS = ('biological_validation', 'living_tissue_connected',
                  'body_simulated', 'walking_demonstrated', 'upstream_lif_executed')


def fingerprint(values):
    a = np.ascontiguousarray(values)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_matrix(value, name):
    a = np.asarray(value)
    require(a.ndim == 2 and a.size > 0 and a.dtype.kind == 'f'
            and np.isfinite(a).all(), f'{name}: finite floating matrix required')
    return a


def close_number(recorded, measured, name):
    if measured is None:
        require(recorded is None, f'{name}: unmeasurable quantity must be null')
    else:
        require(type(recorded) in (int, float) and np.isfinite(recorded)
                and np.isclose(recorded, measured, rtol=1e-9, atol=1e-13),
                f'{name}: recorded metric differs from saved traces')


def metrics(reference, lesion, candidate):
    # Independent implementation: intentionally not importing comparison().
    norm = float(np.sqrt(np.sum(reference * reference)))
    d = lesion - reference
    e = candidate - reference
    damage = float(np.sqrt(np.sum(d * d)))
    error = float(np.sqrt(np.sum(e * e)))
    return {
        'reference_norm': norm, 'lesion_error': damage, 'candidate_error': error,
        'relative_reference_error': error/norm if norm > 1e-12 else None,
        'recovery_fraction': 1-error/damage if damage > max(1e-12, norm*1e-8) else None,
        'maximum_motor_error': float(np.max(np.abs(e))),
    }


def audit_payload(result, execution, traces, source_commit):
    require(result.get('schema') == 'circuit-bridge-malecns/v1', 'Unknown evidence schema')
    require(execution.get('result_present') is True, 'Execution did not produce a result')
    require(isinstance(source_commit, str) and re.fullmatch(r'[0-9a-f]{40}', source_commit),
            'Invalid source commit')
    require(execution.get('source_commit') == source_commit, 'Source commit mismatch')
    require(re.fullmatch(r'[0-9]+', str(execution.get('actions_run_id', ''))) is not None,
            'Missing Actions run ID')
    claims = result['claims']
    require(all(claims.get(key) is False for key in LIMITED_CLAIMS),
            'Evidence cannot support biology/body/living-tissue claims')
    graph = result['graph']
    require(graph.get('neurons') == 166700 and graph.get('connection_rows') == 25582938,
            'Different or partial annotated graph')
    require(graph.get('motor_neurons') == 815, 'Different motor domain')
    parity = graph.get('upstream_byte_parity', {})
    require(set(parity) == {'body_ids.npy', 'edges.bin', 'motor_indices.npy',
                           'glutamate_inhibitory_hypothesis_signs.npy'}
            and all(value is True for value in parity.values()), 'Upstream parity not established')
    exp = result['experiment']
    train_seeds, val_seeds, test_seeds = [exp[key] for key in
        ('train_seeds', 'validation_seeds', 'test_seeds')]
    for name, values in zip(('training','validation','test'), (train_seeds,val_seeds,test_seeds)):
        require(values and all(type(x) is int for x in values) and len(set(values)) == len(values),
                f'{name}: missing or duplicate seeds')
    require(set(train_seeds).isdisjoint(val_seeds) and set(train_seeds).isdisjoint(test_seeds)
            and set(val_seeds).isdisjoint(test_seeds), 'Training/validation/test leakage')
    fitted = traces['training']
    basis = finite_matrix(fitted['basis'], 'basis')
    training = finite_matrix(fitted['training'], 'training')
    region_size, rank = exp['region_size'], exp['selected_rank']
    require(type(rank) is int and 0 < rank <= region_size and basis.shape == (region_size, rank),
            'Basis dimension mismatch')
    require(training.shape[1] == region_size, 'Training region dimension mismatch')
    require(np.allclose(basis.T @ basis, np.eye(rank), rtol=0, atol=1e-10),
            'Nonorthonormal trained basis')
    require(fingerprint(basis) == exp['basis_sha256'], 'Changed trained basis')
    require(fingerprint(training) == exp['training_trace_sha256'], 'Changed training data')
    _, _, vh = np.linalg.svd(training, full_matrices=False)
    reconstructed_basis = vh[:rank].T
    require(np.allclose(basis @ basis.T, reconstructed_basis @ reconstructed_basis.T,
                        rtol=1e-8, atol=1e-9), 'Basis is not the training-only POD subspace')
    candidates = exp['validation']
    require(candidates and all(r['seed'] in val_seeds for r in candidates),
            'Rank selection did not use validation seeds')
    passing = [r['rank'] for r in candidates if r['recovery_fraction'] is not None
               and r['recovery_fraction'] >= .95]
    selected = min(passing) if passing else min(candidates, key=lambda r:r['candidate_error'])['rank']
    require(rank == selected and exp['selection_passed_validation'] is bool(passing),
            'Selected rank disagrees with predeclared validation rule')
    require(fitted['region_ids'].tolist() == exp['region_body_ids'], 'Region IDs/order changed')
    motor_ids = np.asarray(fitted['motor_ids'])
    require(motor_ids.shape == (815,) and motor_ids.dtype.kind in 'iu'
            and len(np.unique(motor_ids)) == 815, 'Motor IDs invalid')
    require(np.asarray(fitted['stimulus_ids']).tolist() == graph['stimulus_body_ids'],
            'Stimulus identity mismatch')
    expected = {(seed, mode) for seed in test_seeds for mode in MODES}
    trials = exp['trials']
    require(len(trials) == len(expected) and {(r['seed'], r['mode']) for r in trials} == expected,
            'Missing/duplicate/unexpected held-out condition')
    by_key = {(r['seed'], r['mode']): r for r in trials}
    rows = []
    all_hashes = {}
    for seed in test_seeds:
        intact, les = traces[f'{seed}-intact'], traces[f'{seed}-lesion']
        ref = finite_matrix(intact['motor'], 'intact motor')
        lesion = finite_matrix(les['motor'], 'lesion motor')
        drive = finite_matrix(intact['drive'], 'intact drive')
        region_ref = finite_matrix(intact['region'], 'intact region')
        require(ref.shape == lesion.shape and ref.shape[1] == 815 and drive.shape == (len(ref),2)
                and region_ref.shape == (len(ref),region_size), 'Reference shape mismatch')
        require(np.count_nonzero(les['region']) == 0, 'Lesioned region was not silenced')
        for mode in MODES:
            data = traces[f'{seed}-{mode}']
            out = finite_matrix(data['motor'], 'candidate motor')
            region = finite_matrix(data['region'], 'candidate region')
            require(out.shape == ref.shape and region.shape == region_ref.shape,
                    'Candidate shape mismatch')
            require(np.array_equal(drive, data['drive']), 'Conditions received different external inputs')
            require(np.max(np.abs(out)) <= 1+1e-12 and np.max(np.abs(region)) <= 1+1e-12,
                    'Output violated dimensionless rate contract')
            row = by_key[seed,mode]
            require(fingerprint(drive) == row['stimulus_sha256'], 'Input hash mismatch')
            require(fingerprint(out) == row['motor_trace_sha256'], 'Motor trace hash mismatch')
            measured = metrics(ref, lesion, out)
            for key, value in measured.items():
                close_number(row[key], value, key)
            if mode == 'native':
                require(np.allclose(ref, out, rtol=1e-10, atol=1e-12)
                        and np.allclose(region_ref, region, rtol=1e-10, atol=1e-12),
                        'Native replacement parity failed')
            else:
                require(row['status'] == ('model_response_only' if measured['recovery_fraction'] is not None
                                         else 'unmeasurable_lesion_effect'), 'Metric status mismatch')
            rows.append({'seed': seed, 'mode': mode, **measured,
                         'lesion_relative_reference_error': measured['lesion_error']/measured['reference_norm']
                             if measured['reference_norm'] > 1e-12 else None})
            all_hashes[f'{seed}-{mode}'] = fingerprint(out)
    aggregate = {}
    for mode in MODES:
        vals = [r['recovery_fraction'] for r in rows if r['mode']==mode]
        finite = [v for v in vals if v is not None]
        aggregate[mode] = {'trials':len(vals), 'measurable_trials':len(finite),
            'recovery_min':min(finite) if finite else None,
            'recovery_max':max(finite) if finite else None,
            'recovery_mean':float(np.mean(finite)) if finite else None}
    return {'schema':'circuit-bridge-evidence-audit/v1', 'verified':True,
        'source_commit':source_commit, 'actions_run_id':str(execution['actions_run_id']),
        'metric_recomputations':len(rows), 'full_annotated_graph':graph['neurons'],
        'region_size':region_size, 'latent_rank':rank, 'aggregate':aggregate,
        'rows':rows, 'motor_trace_hashes':all_hashes,
        'limitations':['Audits saved numerical evidence, not independent re-execution of full graph.',
          'Artificial two-channel drive, one uncalibrated rate model and one anatomical specimen.',
          'Recovery is reduction of deviation from intact model, not restored biological behavior.',
          'POD retains region anatomical operator and all ports; no full-brain or runtime compression claim.']}


def audit_directory(directory):
    d = Path(directory)
    result = json.loads((d/'result.json').read_text())
    execution = json.loads((d/'execution.json').read_text())
    traces = {}
    for path in (d/'traces').glob('*.npz'):
        with np.load(path, allow_pickle=False) as record:
            traces[path.stem] = {key:record[key] for key in record.files}
    return audit_payload(result,execution,traces,(d/'source-commit.txt').read_text().strip())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=audit_directory(a.evidence)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(f"Verified {result['metric_recomputations']} held-out numerical comparisons")


if __name__=='__main__':main()
