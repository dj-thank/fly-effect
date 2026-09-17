"""Post-hoc, fail-closed audit of the pinned full-CNS replacement experiment.

This module reanalyses archived outputs; it does not rerun a brain, demonstrate
walking, or certify biology. Geometry applies only to a fixed linear decoder,
zero initial state, full independent region drive, and all region outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SOURCE_COMMIT = '787b67537034646edb1f61ffac64da188ecbb17d'
RUN_ID = '35189009138'
ARCHIVE_SHA256 = '206577c07a08589dddfe860cc289d89477fe042c7a431d4f9fce4d3f9644ec0a'
RESULT_SHA256 = 'c24b5b659704009787a7d0bbfe435174ba80a6b81a05ae5f7345389399c76d3c'
TRACE_SHA256 = '21821146f22259dc46fe1ca7dba60822069373ab5d8420781462561ccb6b494d'
GAINS = ('0.5', '0.8', '0.95')
COHORTS = {
    'heldout_dn': ((101, 102, 103), ('lesion', 'native', 'pod', 'shuffled', 'random_basis')),
    'heldout_port_stress': ((201, 202, 203), ('lesion', 'native', 'frozen_pod', 'robust_pod', 'robust_random_basis')),
}


def number(x: object) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('Expected a finite real number')
    return float(x)


def integer(x: object, low: int, high: int) -> int:
    if type(x) is not int or not low <= x <= high:
        raise ValueError('Invalid integer')
    return x


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode())
    h.update(memoryview(x).cast('B'))
    return h.hexdigest()


def orthonormal_basis(basis: np.ndarray) -> np.ndarray:
    u = np.asarray(basis, dtype=float)
    if (u.ndim != 2 or not 1 <= u.shape[1] <= u.shape[0]
            or not np.isfinite(u).all()
            or not np.allclose(u.T @ u, np.eye(u.shape[1]), rtol=0, atol=1e-10)):
        raise ValueError('Finite, column-orthonormal basis required')
    return u


def blind_spot(basis: np.ndarray, amplitude: float = .1, alpha: float = .25) -> dict:
    """Construct a bounded first-step input missed by a rank-deficient decoder.

    If U has k<n orthonormal columns, choose unit v orthogonal to its range.
    With c=tanh(A)/||v||_inf and u=atanh(c*v), ||u||_inf<=A. At zero state,
    the native first step is alpha*c*v; every U z has relative L2 error >=1.
    The Galerkin endpoint returns its zero projection. No motor claim follows.
    """
    u = orthonormal_basis(basis)
    amplitude, alpha = number(amplitude), number(alpha)
    if not 0 < amplitude <= 1 or not 0 < alpha <= 1:
        raise ValueError('Require 0<amplitude<=1 and 0<alpha<=1')
    n, rank = u.shape
    if rank == n:
        return {'status': 'full_rank_no_compression', 'n': n, 'rank': rank,
                'nontrivial_compression': False, 'witness': None}
    index = int(np.argmin(np.sum(u * u, axis=1)))
    direction = np.eye(1, n, index).ravel() - u @ u[index]
    direction /= np.linalg.norm(direction)
    c = math.tanh(amplitude) / np.max(np.abs(direction))
    drive = np.arctanh(c * direction)
    native = alpha * np.tanh(drive)
    projection = u @ (u.T @ native)
    relative = float(np.linalg.norm(native - projection) / np.linalg.norm(native))
    if np.max(np.abs(drive)) > amplitude + 1e-12 or abs(relative - 1) > 1e-10:
        raise AssertionError('Bounded orthogonal witness construction failed')
    return {'status': 'bounded_blind_spot_constructed', 'n': n, 'rank': rank,
            'nontrivial_compression': True, 'amplitude': amplitude, 'alpha': alpha,
            'witness': {'drive': drive.tolist(), 'native_next': native.tolist(),
                        'projected_next': projection.tolist(),
                        'relative_region_l2_error': relative,
                        'orthogonality_residual': float(np.linalg.norm(u.T @ direction)),
                        'max_abs_drive': float(np.max(np.abs(drive)))},
            'scope': 'zero-state first step; fixed linear decoder; independent signed region inputs; all region outputs',
            'exclusions': ['not a biological stimulus', 'not a motor-output lower bound',
                           'not a lower bound for nonlinear decoders or full-rank direct feedthrough',
                           'not a novel linear-algebra theorem', 'not a blinded empirical test']}


def isotropic_probe(basis: np.ndarray, samples: int = 8192, seed: int = 20260917,
                    amplitude: float = .1) -> dict:
    u = orthonormal_basis(basis)
    integer(samples, 16, 100000)
    integer(seed, 0, 2**32 - 1)
    amplitude = number(amplitude)
    if not 0 < amplitude <= 1:
        raise ValueError('Invalid probe amplitude')
    drive = np.random.default_rng(seed).uniform(-amplitude, amplitude, (samples, u.shape[0]))
    y = np.tanh(drive)
    residual = y - (y @ u) @ u.T
    # E[||residual||^2] / E[||y||^2] = 1-k/n for iid symmetric components.
    ratio = float(np.sum(residual * residual) / np.sum(y * y))
    return {'samples': samples, 'seed': seed, 'amplitude': amplitude,
            'observed_energy_loss_ratio': ratio,
            'isotropic_population_energy_loss_ratio': 1 - u.shape[1] / u.shape[0],
            'statistical_unit': 'artificial independent input vector, not animal',
            'scope': 'first-step region-state projection; no recurrent or motor simulation'}


def summarize(values: list[float]) -> dict:
    x = [number(v) for v in values]
    if not x:
        raise ValueError('Empty cohort')
    return {'n_conditions': len(x), 'minimum': min(x), 'median': float(np.median(x)),
            'maximum': max(x), 'mean': float(np.mean(x))}


def audit_result(result: dict, threshold: float = .95) -> dict:
    """Require the exact historical cohort; missing/duplicate records are errors.

    The threshold is the earlier benchmark's descriptive criterion. This is
    a retrospective audit, NOT preregistration or a population-level claim.
    """
    threshold = number(threshold)
    if not 0 < threshold <= 1:
        raise ValueError('Invalid threshold')
    if (result.get('schema') != 'circuit-bridge-certified/v1'
            or result.get('status') != 'complete'
            or result.get('source_commit') != SOURCE_COMMIT
            or str(result.get('actions_run_id')) != RUN_ID
            or set(result.get('experiments', {})) != set(GAINS)):
        raise ValueError('Wrong, incomplete, or unpinned historical experiment')
    if any(result.get('claims', {}).get(k) is not False for k in (
            'biological_validation', 'body_simulated', 'living_tissue_connected',
            'walking_demonstrated', 'upstream_lif_executed', 'whole_brain_speedup_demonstrated')):
        raise ValueError('Historical non-biological claims changed or missing')
    report = {'schema': 'circuit-bridge-intervention-audit/v1', 'kind': 'post_hoc_reanalysis',
              'source_commit': SOURCE_COMMIT, 'actions_run_id': RUN_ID,
              'threshold': threshold, 'experiments': {}, 'cohort_count': 0,
              'claims': {'new_full_cns_run': False, 'biological_validation': False,
                         'universal_low_rank_replacement': False}}
    for gain in GAINS:
        exp = result['experiments'][gain]
        n = integer(exp['region_size'], 1, 100000)
        rank = integer(exp['selected_rank'], 1, n)
        robust = integer(exp['stress']['selected_rank'], 1, n)
        splits = [exp['train_seeds'], exp['validation_seeds'], exp['test_seeds'],
                  exp['stress']['train_seeds'], exp['stress']['validation_seeds'],
                  exp['stress']['test_seeds']]
        flat = [integer(s, 0, 2**32 - 1) for split in splits for s in split]
        if len(flat) != len(set(flat)):
            raise ValueError('Overlapping or duplicate train/validation/test seeds')
        expected = {(c, s, m) for c, (seeds, modes) in COHORTS.items() for s in seeds for m in modes}
        rows = {}
        for row in exp['trials']:
            key = (row['cohort'], integer(row['seed'], 0, 2**32 - 1), row['mode'])
            if key not in expected or key in rows:
                raise ValueError('Unexpected or duplicate historical trial')
            error, damage = number(row['candidate_error']), number(row['lesion_error'])
            if error < 0 or damage <= 0:
                raise ValueError('Invalid or unmeasurable intervention effect')
            recovery = number(row['recovery_fraction'])
            if not math.isclose(recovery, 1 - error / damage, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError('Reported recovery disagrees with raw norms')
            if row['mode'] == 'native' and error > 1e-10:
                raise ValueError('Native endpoint parity failed')
            rows[key] = row
        if set(rows) != expected:
            raise ValueError('Missing historical trials: do not silently omit failures')
        cohorts = {}
        for cohort, (_, modes) in COHORTS.items():
            cohorts[cohort] = {mode: summarize([v['recovery_fraction'] for k, v in rows.items()
                                              if k[0] == cohort and k[2] == mode]) for mode in modes}
        dn = cohorts['heldout_dn']['pod']
        stress = cohorts['heldout_port_stress']['frozen_pod']
        robust_metrics = cohorts['heldout_port_stress']['robust_pod']
        flags = {
            'nontrivial_compression_on_dn_inputs': rank < n and dn['minimum'] >= threshold,
            'frozen_basis_passes_independent_port_stress': stress['minimum'] >= threshold,
            'robust_model_is_nontrivial_compression': robust < n,
            'robust_nontrivial_stress_success': robust < n and robust_metrics['minimum'] >= threshold,
            'shuffled_control_also_passes_dn_threshold': cohorts['heldout_dn']['shuffled']['minimum'] >= threshold,
        }
        report['experiments'][gain] = {'region_size': n, 'rank_dn': rank, 'rank_stress': robust,
                                      'cohorts': cohorts, 'gates': flags}
        report['cohort_count'] += len(rows)
    report['interpretation_limits'] = [
        'Seeds are stimulus conditions from one anatomical specimen, not independent animals.',
        'The existing stress comparison changes input ports and amplitude together; attribution is confounded.',
        'A full-rank successful substitute is a parity control, not a compression success.',
        'Large shuffled-control recovery means this assay does not establish unique topology necessity.',
        'Global motor-trace norms do not establish per-motor or behavioral equivalence.',
        'A numerical error bound for a contractive rate hypothesis is not biological validation.',
    ]
    return report


def motor_channel_audit(reference, lesion, candidate, ids, threshold=.95):
    """Post-hoc per-output sensitivity analysis, not a biological pass/fail test.

    Report all three effect floors rather than choose a favorable denominator.
    Body IDs are outputs of the archived model, not experimental recordings.
    """
    a, b, c = [np.asarray(x, dtype=float) for x in (reference, lesion, candidate)]
    ids = np.asarray(ids)
    if (a.ndim != 2 or not a.size or a.shape != b.shape or a.shape != c.shape
            or ids.shape != (a.shape[1],) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids)
            or not all(np.isfinite(x).all() for x in (a, b, c))):
        raise ValueError('Finite aligned traces and unique integer motor IDs required')
    threshold = number(threshold)
    if not 0 < threshold <= 1:
        raise ValueError('Invalid descriptive threshold')
    damage = np.linalg.norm(b-a, axis=0)
    error = np.linalg.norm(c-a, axis=0)
    maximum = float(damage.max())
    floors = []
    for fraction in (.001, .01, .1):
        floor = max(1e-12, fraction*maximum)
        mask = damage >= floor
        if not mask.any():
            floors.append({'relative_effect_floor': fraction, 'eligible': 0,
                           'status': 'unmeasurable_lesion_effect'})
            continue
        value = 1-error[mask]/damage[mask]
        worst = int(np.argmin(value))
        floors.append({'relative_effect_floor': fraction, 'absolute_effect_floor': floor,
                       'eligible': int(mask.sum()), 'below_threshold': int((value<threshold).sum()),
                       'minimum': float(value.min()), 'median': float(np.median(value)),
                       'worst_motor_body_id': int(ids[mask][worst]),
                       'status': 'post_hoc_model_output_diagnostic'})
    return {'metric': 'one minus per-channel error norm divided by lesion norm',
            'threshold': threshold, 'effect_floor_sensitivity': floors,
            'scope': 'archived dimensionless motor-model outputs; not measured biological recovery'}


def audit_traces(result: dict, archive: Path) -> dict:
    if file_hash(archive) != TRACE_SHA256:
        raise ValueError('Pinned trace archive checksum mismatch')
    exp = result['experiments']['0.8']
    with np.load(archive, allow_pickle=False) as z:
        for key, expected in [('basis', exp['basis_sha256']), ('training_region', exp['training_trace_sha256']),
                              ('stress_basis', exp['stress']['basis_sha256'])]:
            if array_hash(z[key]) != expected:
                raise ValueError('Trace/basis identity mismatch')
        basis = z['basis']
        _, _, vh = np.linalg.svd(z['training_region'], full_matrices=False)
        expected = vh[:basis.shape[1]].T
        if not np.allclose(basis @ basis.T, expected @ expected.T, atol=1e-10, rtol=0):
            raise ValueError('Basis is not the training-trace leading subspace')
        trace_checks = []
        for row in exp['trials']:
            prefix = f"g0.8_{row['cohort']}_{row['seed']}"
            reference, candidate = z[prefix + '_reference'], z[row['trace_key']]
            if array_hash(candidate) != row['motor_trace_sha256']:
                raise ValueError('Motor trace hash mismatch')
            error = float(np.linalg.norm(candidate - reference))
            if not math.isclose(error, row['candidate_error'], abs_tol=1e-10, rel_tol=1e-9):
                raise ValueError('Archived error disagrees with trace recomputation')
            bound = z[row['trace_key'] + '_bound']
            violation = float(np.max(np.max(np.abs(candidate-reference), axis=1) - bound))
            if violation > 1e-12:
                raise ValueError('Motor trace violates declared numerical bound')
            trace_checks.append({'cohort': row['cohort'], 'seed': row['seed'], 'mode': row['mode'],
                                 'error_recomputed': error, 'max_bound_violation': violation,
                                 'per_motor': motor_channel_audit(reference, z[prefix+'_lesion'],
                                                                  candidate, z['motor_body_ids'])})
        return {'trace_sha256': TRACE_SHA256, 'checks': trace_checks,
                'basis_recomputed_from_training_only': True,
                'geometry': blind_spot(basis), 'isotropic_probe': isotropic_probe(basis),
                'full_rank_geometry': blind_spot(z['stress_basis'])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite an existing analysis')
    if file_hash(args.result) != RESULT_SHA256:
        raise ValueError('Pinned historical result checksum mismatch')
    result = json.loads(args.result.read_text())
    report = audit_result(result)
    report['source_result_sha256'] = file_hash(args.result)
    report['trace_analysis'] = audit_traces(result, args.traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'historical_trial_records_audited': report['cohort_count'],
                      'trace_records_recomputed': len(report['trace_analysis']['checks']),
                      'new_full_cns_run': False}))


if __name__ == '__main__':
    main()
