"""Explain fixed-pose passive-load conflicts without changing model parameters.

Row deletion yields ONE inclusion-minimal inconsistent subsystem relative to
retained root equations, NOT a minimum-cardinality or unique biological cause.
Every call uses the existing support solver, forces, friction and tolerances.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from .static_support import solve_support

FIELDS = ('support_tendon_map', 'support_contact_map', 'support_target',
          'support_limits', 'support_friction')
INFEASIBLE = 'infeasible_under_declared_constraints'
PROTOCOL = {'id': 'FE-02-passive-conflict-v1', 'order': 'ascending_exact_zero_rows',
            'wall_seconds': 120., 'max_calls_per_pose': 128,
            'single_row_checks': True, 'final_deletions_rechecked': True,
            'solver': 'unchanged static_support.solve_support',
            'scope': 'fixed-pose necessary conditions; not a new physical trajectory'}


class UnresolvedSupport(RuntimeError):
    """A solver did not establish feasibility or declared infeasibility."""


def _positive(value, name, *, integer=False):
    if ((type(value) is not int if integer else type(value) not in (int, float))
            or value <= 0):
        raise ValueError('Positive ' + name + ' required')
    try:
        if not np.isfinite(float(value)):
            raise ValueError('Finite ' + name + ' required')
    except OverflowError as error:
        raise ValueError('Finite ' + name + ' required') from error


def diagnose(tendon_map, contact_map, target, maximum_tension, friction, *,
             root_rows=(0, 1, 2, 3, 4, 5), wall_seconds=120., max_calls=128):
    """Keep all force columns and exact nonzero values, including root roundoff.

    Deadlines are checked between calls. An already running original LP can use
    up to its existing time limit; use an outer supervisor for a hard deadline.
    A timeout/numerical failure never becomes a mechanical infeasibility claim.
    """
    _positive(wall_seconds, 'wall_seconds'); _positive(max_calls, 'max_calls', integer=True)
    deadline = time.monotonic() + wall_seconds
    T, C, b, limit, mu = (np.asarray(v, dtype=float) for v in
                           (tendon_map, contact_map, target, maximum_tension, friction))
    if (T.ndim != 2 or C.ndim != 2 or T.shape[0] != C.shape[0] or not T.size
            or C.shape[1] == 0 or C.shape[1] % 3 or b.shape != (T.shape[0],)
            or limit.shape != (T.shape[1],) or mu.shape != (C.shape[1]//3,)):
        raise ValueError('Incompatible support dimensions')
    if not all(np.isfinite(x).all() for x in (T, C, b, limit, mu)) or np.any(limit <= 0) or np.any(mu < 0):
        raise ValueError('Finite support data, positive limits and nonnegative friction required')
    roots = list(root_rows)
    if (not roots or any(type(i) is not int or i < 0 or i >= len(b) for i in roots)
            or len(set(roots)) != len(roots)):
        raise ValueError('Unique in-range integer root rows required')
    passive = [int(i) for i in np.flatnonzero(np.all(T == 0, axis=1)) if i not in roots]
    result = {'status': 'inconclusive', 'root_rows': roots, 'passive_rows': passive,
              'checks': [], 'single_rows': [], 'conflict_rows': [],
              'inclusion_minimal': False, 'minimum_cardinality_claimed': False,
              'unique_cause_claimed': False, 'full_support_claimed': False,
              'walking_claimed': False, 'standing_claimed': False, 'biological_validation': False}
    cache = {}

    def check(extra, *, fresh=False):
        if time.monotonic() >= deadline:
            raise TimeoutError('Passive-conflict wall budget exceeded')
        rows = tuple(roots + sorted(extra))
        if not fresh and rows in cache:
            return cache[rows]
        if len(result['checks']) >= max_calls:
            raise TimeoutError('Passive-conflict solver-call budget exceeded')
        ids = np.asarray(rows, dtype=int)
        solved = solve_support(T[ids], C[ids], b[ids], limit, mu)
        result['checks'].append({'rows': list(rows), 'report': solved})
        if time.monotonic() >= deadline:
            raise TimeoutError('Passive-conflict wall budget exceeded after solve')
        status = solved.get('status')
        if status not in ('feasible', INFEASIBLE):
            raise UnresolvedSupport('Support subproblem unresolved: ' + str(status))
        expected = status == 'feasible'
        if (solved.get('feasible') is not expected or solved.get('solver_success') is not expected
                or type(solved.get('solver_status')) is not int
                or solved['solver_status'] != (0 if expected else 2)):
            raise ValueError('Contradictory solver feasibility or status')
        cache[rows] = status
        return status

    try:
        result['root_status'] = check([])
        if result['root_status'] != 'feasible':
            result['status'] = 'root_balance_unresolved'
            return result
        result['passive_projection_status'] = check(passive)
        if result['passive_projection_status'] == 'feasible':
            result['status'] = 'no_passive_conflict_found'
            return result  # Other/actuated equations have not been certified.
        for row in passive:
            result['single_rows'].append({'row': row, 'status': check([row])})
        retained = passive.copy()
        for row in passive:
            trial = [i for i in retained if i != row]
            if check(trial) == INFEASIBLE:
                retained = trial
        # Fresh final checks prevent caching alone from certifying irreducibility.
        if check(retained, fresh=True) != INFEASIBLE:
            raise UnresolvedSupport('Final conflict did not reproduce')
        deletion_checks = []
        for row in retained:
            status = check([i for i in retained if i != row], fresh=True)
            deletion_checks.append({'removed_row': row, 'status': status})
            if status != 'feasible':
                raise UnresolvedSupport('Final row deletion did not reproduce feasibility')
        result.update(status='passive_conflict_isolated', conflict_rows=retained,
                      inclusion_minimal=True, final_deletion_checks=deletion_checks,
                      solver_calls=len(result['checks']))
        return result
    except Exception as error:
        result.update(status='inconclusive' if isinstance(error, (TimeoutError, UnresolvedSupport)) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        error.diagnostic = result
        raise
    finally:
        result['solver_calls'] = len(result['checks'])


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(workspace):
    """Reverify retained inputs and write a separate, non-overwriting diagnostic."""
    from .foot_placement import verify
    from .root_ab import write_json
    from .root_ab_evidence import read_json
    workspace = Path(workspace).resolve()
    out = workspace/'passive-conflict'; out.mkdir(exist_ok=False)
    source = workspace/'foot-placement'; started = time.monotonic()
    inputs = [source/'result.json', source/'observations.npz', source/'model-receipt.json',
              workspace/'protocol.json', workspace/'source-study/result.json',
              workspace/'source-study/observations.npz']
    record = {'schema': 1, 'protocol': PROTOCOL, 'status': 'running', 'candidates': [],
              'biological_validation': False, 'walking_claimed': False, 'standing_claimed': False,
              'CNS_executed': False, 'new_physics_trajectories': 0}
    try:
        record['input_sha256'] = {str(p.relative_to(workspace)): _hash(p) for p in inputs}
        record['code_sha256'] = {name: _hash(Path(__file__).with_name(name)) for name in
                                ('support_conflict.py', 'static_support.py', 'foot_placement.py', 'root_ab_evidence.py')}
        record['versions'] = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy')}
        write_json(out/'protocol.json', record)
        # Refreshes the ordinary verification receipt, never physical source inputs.
        record['source_verification'] = verify(workspace)
        if record['source_verification'].get('evidence_valid') is not True:
            raise ValueError('Source evidence was not verified')
        result = read_json(source/'result.json')
        with np.load(source/'observations.npz', allow_pickle=False) as arrays:
            for candidate in result['candidates']:
                if 'support' not in candidate:
                    continue
                remaining = PROTOCOL['wall_seconds'] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('Passive-conflict overall wall budget exceeded')
                index = candidate['index']; prefix = f'pose_{index:02d}_'
                try:
                    diagnosis = diagnose(*(arrays[prefix+k] for k in FIELDS),
                                         root_rows=candidate['force_accounting']['root_dofs'],
                                         wall_seconds=remaining, max_calls=PROTOCOL['max_calls_per_pose'])
                except Exception as error:
                    if hasattr(error, 'diagnostic'):
                        record['candidates'].append({'index': index, **error.diagnostic})
                    raise
                record['candidates'].append({'index': index, **diagnosis})
        after = {str(p.relative_to(workspace)): _hash(p) for p in inputs}
        if after != record['input_sha256']:
            raise ValueError('Physical source evidence changed during diagnosis')
        if time.monotonic() - started >= PROTOCOL['wall_seconds']:
            raise TimeoutError('Passive-conflict overall wall budget exceeded after solve')
        record.update(status='completed', physical_inputs_unchanged=True,
                      candidates_checked=len(record['candidates']),
                      conflicts_isolated=sum(c['status'] == 'passive_conflict_isolated' for c in record['candidates']))
    except Exception as error:
        record.update(status='inconclusive' if isinstance(error, (TimeoutError, UnresolvedSupport)) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['wall_seconds'] = time.monotonic() - started
        write_json(out/'result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.workspace)
    print(json.dumps({k: v for k, v in report.items() if k != 'candidates'}, indent=2))
