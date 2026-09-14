"""Combined passive spring-authority margin over saved support evidence (FE-02).

For every retained support witness, this asks whether ALL passive rows' spring
shifts TOGETHER — each inside its achievable interval k_d*(range_d - q_d) —
could satisfy the root+passive-row subsystem at the saved contact geometry.
Unlike the per-row screen in `passive_posture.py`, rows move simultaneously
here, and the solver returns a single authority fraction t: t<=1 means the
conflict lies inside combined passive authority at this fixed geometry;
t>1 means the required shift needs t times the achievable box.

This is a fixed-geometry screen, not physical evidence: moving a joint also
changes the contact map and bias, so a large t neither proves nor disproves
what other poses could do. It exists to aim — or rule out — posture
generation. No dynamics, standing, walking or biological claim follows.

Inputs are only saved witness arrays; nothing physical is re-derived and no
model parameter is touched. An undecided solver state is never turned into a
physical infeasibility claim.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
import numpy as np

INFEASIBLE = 'infeasible_under_declared_constraints'
FIELDS = ('support_tendon_map', 'support_contact_map', 'support_target',
          'support_limits', 'support_friction')
WITNESS = re.compile(r'^(pose_\d+_trial_\d+_depth_\d+_)support_tendon_map$')
PROTOCOL = {
    'id': 'FE-02-passive-margin-v1',
    'margin_variable': ('uniform scale t of every achievable passive spring '
                        'shift interval k_d*(range_d - q_d); t<=1 means within '
                        'combined authority at this fixed geometry'),
    'authority_interval': 'full closed joint interval, no inset',
    'rows': 'six free-root equations plus every exactly-zero tendon-map row',
    'solver': 'scipy linprog highs with the unchanged scaling and tolerances',
    'time_limit_s': 10., 'max_solver_calls': 512, 'wall_seconds': 240.,
    'replay_tolerance': 1e-6,
    'outcomes': ['within_combined_authority', 'beyond_combined_authority',
                 'inconclusive', 'invalid_experiment'],
    'inputs': 'saved foot-placement and passive-posture witnesses only; no physics',
    'scope': 'fixed-geometry combined-authority screen; moving joints also moves '
             'contact geometry and bias, so margins are local to each saved pose',
    'dynamic_trials': 0, 'biological_validation': False,
}


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _positive(value, name, *, integer=False):
    if ((type(value) is not int if integer else type(value) not in (int, float))
            or value <= 0 or not np.isfinite(float(value))):
        raise ValueError('Positive finite ' + name + ' required')


def passive_rows(tendon_map, root_rows):
    T = np.asarray(tendon_map, dtype=float)
    roots = list(root_rows)
    return [int(i) for i in np.flatnonzero(np.all(T == 0, axis=1)) if i not in roots]


def achievable_intervals(stiffness, ranges, q_current):
    """[lo,hi] of the spring-torque shift k*(q_new - q_now) per passive DOF."""
    k = np.asarray(stiffness, dtype=float)
    r = np.asarray(ranges, dtype=float)
    q = np.asarray(q_current, dtype=float)
    if (k.ndim != 1 or r.shape != (len(k), 2) or q.shape != (len(k),)
            or not np.isfinite(k).all() or not np.isfinite(r).all()
            or not np.isfinite(q).all() or np.any(r[:, 0] >= r[:, 1])
            or np.any(q < r[:, 0]) or np.any(q > r[:, 1]) or np.any(k < 0)):
        raise ValueError('Invalid passive spring inputs')
    return np.column_stack((k*(r[:, 0]-q), k*(r[:, 1]-q)))


def margin_solve(tendon_map, contact_map, target, limits, friction,
                 root_rows, passive, intervals):
    """Min t such that (C r)_d - b_d lies inside t*interval_d for all passive rows.

    Same unknowns and constraints as the unchanged support LP plus one scalar t.
    Status 2 means no finite authority scaling suffices — the needed residual
    direction is unreachable by passive springs at this geometry and sign.
    """
    from scipy.optimize import linprog
    T, C = np.asarray(tendon_map, dtype=float), np.asarray(contact_map, dtype=float)
    b, limit, mu = (np.asarray(x, dtype=float) for x in (target, limits, friction))
    I = np.asarray(intervals, dtype=float)
    roots, pas = list(root_rows), list(passive)
    if (T.ndim != 2 or C.ndim != 2 or T.shape[0] != C.shape[0] or not T.shape[0]
            or not T.shape[1] or C.shape[1] % 3 or not C.shape[1]
            or b.shape != (T.shape[0],) or limit.shape != (T.shape[1],)
            or mu.shape != (C.shape[1]//3,) or I.shape != (len(pas), 2)
            or not pas or len(set(pas)) != len(pas) or len(set(roots)) != len(roots)
            or set(pas) & set(roots)):
        raise ValueError('Incompatible margin dimensions')
    if (not all(np.isfinite(x).all() for x in (T, C, b, limit, mu, I))
            or np.any(limit <= 0) or np.any(mu < 0) or np.any(I[:, 0] > I[:, 1])):
        raise ValueError('Finite margin data and ordered intervals required')
    if np.any(T[pas] != 0):
        raise ValueError('Declared passive rows must have exactly zero tendon entries')
    n, k3 = len(limit), C.shape[1]
    m = n+k3+1    # variables: tensions, contact forces, authority fraction t
    scale = np.maximum(np.abs(b), np.max(np.abs(T)*limit, axis=1))
    scale = np.maximum(scale, 1e-6)
    A_eq = np.column_stack((T[roots], C[roots], np.zeros(len(roots))))/scale[roots, None]
    b_eq = b[roots]/scale[roots]
    k = k3//3
    ineq = np.zeros((4*k+2*len(pas), m))
    for j in range(k):
        for row, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, 1)) if False
                                       else ((1, 1), (1, -1), (-1, 1), (-1, -1))):
            ineq[4*j+row, n+3*j:n+3*j+3] = [sx, sy, -mu[j]]
    for i, d in enumerate(pas):
        hi, lo = I[i, 1], I[i, 0]
        ineq[4*k+2*i, n:n+3*k] = C[d]/scale[d]
        ineq[4*k+2*i, -1] = -hi/scale[d]
        ineq[4*k+2*i+1, n:n+3*k] = -C[d]/scale[d]
        ineq[4*k+2*i+1, -1] = lo/scale[d]
    b_ub = np.zeros(4*k+2*len(pas))
    b_ub[4*k::2] = b[pas]/scale[pas]
    b_ub[4*k+1::2] = -b[pas]/scale[pas]
    bounds = [(0., float(x)) for x in limit] + [(None, None), (None, None), (0., None)]*k
    bounds.append((0., None))
    cost = np.zeros(m); cost[-1] = 1.
    result = linprog(cost, A_ub=ineq, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs',
                     options={'time_limit': PROTOCOL['time_limit_s']})
    report = {'kind': 'combined_passive_authority_margin', 'solver_status': int(result.status),
              'solver_message': str(result.message), 'solver_success': bool(result.success),
              'status': 'inconclusive', 'biological_validation': False,
              'scope': 'Fixed-geometry combined-authority screen on one saved witness; '
                       'not a physical posture and not full actuated support.'}
    if result.success:
        x = np.asarray(result.x, dtype=float)
        if x.shape != (m,) or not np.isfinite(x).all():
            report['status'] = 'invalid_solver_solution'
            return report
        forces = x[n:n+3*k].reshape(k, 3)
        root_res = np.column_stack((T[roots], C[roots]))@x[:n+3*k] - b[roots]
        passive_res = C[pas]@x[n:n+3*k] - b[pas]
        interval_ok = all(x[-1]*lo-1e-9*max(1., abs(lo)) <= delta
                          <= x[-1]*hi+1e-9*max(1., abs(hi))
                          for delta, (lo, hi) in zip(passive_res, I))
        violations = [float(np.max(np.abs(root_res)/scale[roots])),
                      float(max(0., np.max(-x[:n]/limit))),
                      float(max(0., np.max(x[:n]/limit-1))),
                      float(max(0., np.max(-forces[:, 2]))),
                      float(max(0., np.max(ineq@x-b_ub))),
                      0. if x[-1] >= 0. else float(-x[-1])]
        report.update(t=float(x[-1]),
                      required_shift_native=passive_res.tolist(),
                      shift_fraction_of_interval=[float(passive_res[i]/max(abs(I[i,0]), abs(I[i,1]), 1e-30))
                                                  for i in range(len(pas))],
                      maximum_scaled_root_residual=violations[0],
                      maximum_constraint_violation=max(violations[1:]),
                      tensions_native=x[:n].tolist(),
                      contact_forces_native=forces.tolist(),
                      interval_respected=bool(interval_ok),
                      feasible_within_authority=bool(x[-1] <= 1.+1e-9),
                      solver_checks_passed=all(v <= 1e-6 for v in violations))
        report['status'] = 'measured' if report['solver_checks_passed'] else 'residual_check_failed'
    elif result.status == 2:
        report['status'] = 'unreachable_at_any_shift'
    return report


def _witness_sets(arrays):
    """All saved support-witness prefixes present in an observations mapping."""
    out = []
    for key in arrays:
        match = WITNESS.match(key)
        if match and all(match.group(1)+f in arrays for f in FIELDS):
            out.append(match.group(1))
    return sorted(set(out))


def run(workspace):
    """Measure the combined-authority margin on every retained witness."""
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import read_json
    from .passive_posture import passive_rows as _rows, PROTOCOL as _pp
    del _pp
    workspace = Path(workspace).resolve()
    out = workspace/'passive-margin'; out.mkdir(exist_ok=False)
    placed_dir = workspace/'foot-placement'; posture_dir = workspace/'passive-posture'
    started = time.monotonic(); deadline = started+PROTOCOL['wall_seconds']
    record = {'schema': 1, 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
              'status': 'running', 'candidates': [], 'biological_validation': False,
              'walking_claimed': False, 'standing_claimed': False,
              'CNS_executed': False, 'dynamic_trials_executed': 0,
              'new_physics_trajectories': 0}
    arrays = {}
    try:
        if not posture_dir.is_dir():
            raise FileNotFoundError('passive-margin requires retained passive-posture evidence')
        posture_verdict = read_json(posture_dir/'verification.json')
        if posture_verdict.get('evidence_valid') is not True:
            raise ValueError('Passive-posture evidence was not verified')
        inputs = [placed_dir/'result.json', placed_dir/'observations.npz',
                  placed_dir/'model-receipt.json', workspace/'protocol.json',
                  posture_dir/'result.json', posture_dir/'observations.npz',
                  posture_dir/'verification.json',
                  workspace/'source-study/result.json', workspace/'source-study/observations.npz']
        record['input_sha256'] = {p.relative_to(workspace).as_posix(): file_hash(p) for p in inputs}
        record['code_sha256'] = {name: file_hash(Path(__file__).with_name(name)) for name in
                                 ('passive_margin.py', 'static_support.py', 'foot_placement.py',
                                  'passive_posture.py', 'root_ab_evidence.py')}
        record['versions'] = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy')}
        write_json(out/'protocol.json', record)
        placed = read_json(placed_dir/'result.json')
        posture = read_json(posture_dir/'result.json')
        if posture.get('protocol', {}).get('id') != 'FE-02-passive-posture-v2':
            raise ValueError('Expected FE-02-passive-posture-v2 evidence')
        with np.load(placed_dir/'observations.npz', allow_pickle=False) as z:
            p_arrays = {k: z[k].copy() for k in z.files}
        with np.load(posture_dir/'observations.npz', allow_pickle=False) as z:
            t_arrays = {k: z[k].copy() for k in z.files}
        calls = 0
        for entry in posture['candidates']:
            index = entry['index']; prefix = f'pose_{index:02d}_'
            candidate = placed['candidates'][index]
            if candidate['index'] != index:
                raise ValueError('Candidate ordering changed between experiments')
            if 'support' not in candidate:
                continue
            passive = [int(i) for i in t_arrays[prefix+'passive_dofs']]
            qpos_adrs = t_arrays[prefix+'passive_qpos_adrs']
            stiffness = t_arrays[prefix+'passive_stiffness']
            ranges = t_arrays[prefix+'passive_ranges']
            root_rows = candidate['force_accounting']['root_dofs']
            if passive != _rows(p_arrays[prefix+'support_tendon_map'], root_rows):
                raise ValueError('Passive row set differs between saved matrices')
            witnesses = [{'source': 'foot-placement', 'prefix': prefix,
                          'arrays': p_arrays}]
            for wp in _witness_sets(t_arrays):
                if wp.startswith(prefix):
                    meta = wp[len(prefix):].split('_')
                    witnesses.append({'source': 'passive-posture', 'prefix': wp,
                                      'trial': int(meta[1]), 'depth': int(meta[3]),
                                      'arrays': t_arrays})
            cand = {'index': index, 'witnesses': [], 'solver_calls': 0,
                    'status': 'completed'}
            for w in witnesses:
                if calls >= PROTOCOL['max_solver_calls']:
                    raise TimeoutError('Passive-margin solver-call budget exceeded')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Passive-margin wall budget exceeded')
                a = w['arrays']; wp = w['prefix']
                q_support = a[wp+'support_qpos']
                intervals = achievable_intervals(stiffness, ranges, q_support[qpos_adrs])
                report = margin_solve(a[wp+'support_tendon_map'], a[wp+'support_contact_map'],
                                      a[wp+'support_target'], a[wp+'support_limits'],
                                      a[wp+'support_friction'], root_rows, passive, intervals)
                calls += 1; cand['solver_calls'] += 1
                name = f'w_{len(arrays):04d}_'
                arrays[name+'t'] = np.asarray([report.get('t', np.nan)])
                arrays[name+'required_shift'] = np.asarray(
                    report.get('required_shift_native',
                               np.full(len(passive), np.nan)), dtype=float)
                arrays[name+'intervals'] = intervals
                keep = {'witness': name[:-1], 'source': w['source'], 'prefix': wp,
                        'intervals': intervals.tolist()}
                for key in ('trial', 'depth'):
                    if key in w:
                        keep[key] = w[key]
                keep.update({k: report[k] for k in
                             ('status', 'solver_status', 'solver_message', 'scope')
                             if k in report})
                if report['status'] == 'measured':
                    keep.update(t=report['t'],
                                feasible_within_authority=report['feasible_within_authority'],
                                maximum_scaled_root_residual=report['maximum_scaled_root_residual'],
                                maximum_constraint_violation=report['maximum_constraint_violation'],
                                interval_respected=report['interval_respected'])
                cand['witnesses'].append(keep)
            measured = [w for w in cand['witnesses'] if w['status'] == 'measured']
            cand['minimum_t'] = min((w['t'] for w in measured), default=None)
            cand['within_authority_witnesses'] = sum(
                w['feasible_within_authority'] for w in measured)
            cand['unresolved_witnesses'] = sum(
                w['status'] == 'inconclusive' for w in cand['witnesses'])
            record['candidates'].append(cand)
        after = {p.relative_to(workspace).as_posix(): file_hash(p) for p in inputs}
        if after != record['input_sha256']:
            raise ValueError('Physical source evidence changed during the margin run')
        measured_all = [w for c in record['candidates'] for w in c['witnesses']
                        if w['status'] == 'measured']
        if any(w['status'] == 'inconclusive' for c in record['candidates']
               for w in c['witnesses']) or not measured_all:
            outcome = 'inconclusive'
        elif any(w['feasible_within_authority'] for w in measured_all):
            outcome = 'within_combined_authority'
        else:
            outcome = 'beyond_combined_authority'
        record.update(status='completed', scientific_outcome=outcome,
                      solver_calls=calls,
                      witnesses_checked=len(measured_all)
                      + sum(c['unresolved_witnesses'] for c in record['candidates']),
                      minimum_t=min((w['t'] for w in measured_all), default=None))
    except Exception as exc:
        record.update(status='inconclusive' if isinstance(exc, TimeoutError) else 'failed',
                      scientific_outcome='inconclusive' if isinstance(exc, TimeoutError)
                      else 'invalid_experiment',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if arrays:
            np.savez_compressed(out/'observations.npz', **arrays)
            record['observations'] = {'file': 'observations.npz',
                                      'sha256': file_hash(out/'observations.npz')}
        record['wall_seconds'] = time.monotonic()-started
        write_json(out/'result.json', record)
    return record


def verify(workspace):
    """Replay every saved margin LP from retained arrays; fail closed on mismatch."""
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import read_json, require
    from .passive_posture import passive_rows as _rows
    workspace = Path(workspace); out = workspace/'passive-margin'
    verdict = {'status': 'failed', 'evidence_valid': False,
               'scientific_outcome': 'invalid_experiment', 'walking_claimed': False,
               'standing_claimed': False, 'biological_validation': False}
    try:
        result = read_json(out/'result.json')
        require(all(result.get(k) is False for k in ('walking_claimed', 'standing_claimed',
                'CNS_executed', 'biological_validation'))
                and result.get('dynamic_trials_executed', 0) == 0
                and result.get('new_physics_trajectories') == 0,
                'unsupported capability claim')
        require(result.get('protocol') == PROTOCOL
                and result.get('protocol_sha256') == protocol_hash(), 'protocol mismatch')
        frozen = read_json(out/'protocol.json')
        require(frozen.get('protocol') == PROTOCOL
                and frozen.get('input_sha256') == result.get('input_sha256'),
                'pre-execution protocol record mismatch')
        for rel, recorded in (result.get('input_sha256') or {}).items():
            require(file_hash(workspace/rel) == recorded, 'input evidence changed: '+rel)
        require(result['status'] == 'completed'
                and result.get('observations', {}).get('sha256')
                == file_hash(out/'observations.npz'), 'incomplete result')
        with np.load(out/'observations.npz', allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite saved observations')
        placed = read_json(workspace/'foot-placement/result.json')
        posture = read_json(workspace/'passive-posture/result.json')
        with np.load(workspace/'foot-placement/observations.npz', allow_pickle=False) as z:
            p_arrays = {k: z[k].copy() for k in z.files}
        with np.load(workspace/'passive-posture/observations.npz', allow_pickle=False) as z:
            t_arrays = {k: z[k].copy() for k in z.files}
        all_witnesses = _witness_sets(t_arrays)
        checked = unresolved = 0
        within_any = False
        minima = []
        for cand in result['candidates']:
            index = cand['index']; prefix = f'pose_{index:02d}_'
            candidate = placed['candidates'][index]
            require(candidate['index'] == index, 'candidate ordering changed')
            passive = [int(i) for i in t_arrays[prefix+'passive_dofs']]
            qpos_adrs = t_arrays[prefix+'passive_qpos_adrs']
            stiffness = t_arrays[prefix+'passive_stiffness']
            ranges = t_arrays[prefix+'passive_ranges']
            root_rows = candidate['force_accounting']['root_dofs']
            require(passive == _rows(p_arrays[prefix+'support_tendon_map'], root_rows),
                    'saved passive rows are not the exact-zero rows')
            seen = set()
            for w in cand['witnesses']:
                name = w['witness']+'_'
                require(w['prefix']+ 'support_tendon_map' in
                        (p_arrays if w['source'] == 'foot-placement' else t_arrays),
                        'witness prefix missing from declared source')
                a = p_arrays if w['source'] == 'foot-placement' else t_arrays
                wp = w['prefix']; seen.add(wp)
                if w['source'] == 'passive-posture':
                    meta = wp[len(prefix):].split('_')
                    entry = next(e for e in posture['candidates'] if e['index'] == index)
                    trial = entry['trials'][int(meta[1])]
                    sample = trial['samples'][int(meta[3])]
                    require(sample['index'] == int(meta[3])
                            and sample['evidence_saved'] is True,
                            'witness not marked evidence_saved in posture record')
                require(name+'t' in arrays and name+'required_shift' in arrays
                        and name+'intervals' in arrays, 'missing saved margin arrays')
                intervals = achievable_intervals(stiffness, ranges,
                                                 a[wp+'support_qpos'][qpos_adrs])
                np.testing.assert_allclose(arrays[name+'intervals'], intervals,
                                           rtol=0, atol=0)
                np.testing.assert_allclose(w['intervals'], intervals, rtol=0, atol=0)
                replay = margin_solve(a[wp+'support_tendon_map'], a[wp+'support_contact_map'],
                                      a[wp+'support_target'], a[wp+'support_limits'],
                                      a[wp+'support_friction'], root_rows, passive, intervals)
                require(replay['status'] == w['status']
                        and replay['solver_status'] == w['solver_status'],
                        'margin solve does not reproduce')
                if w['status'] != 'measured':
                    unresolved += int(w['status'] == 'inconclusive')
                    continue
                checked += 1
                tol = PROTOCOL['replay_tolerance']
                require(abs(replay['t']-w['t']) <= tol*max(1., w['t']),
                        'margin t does not reproduce')
                require(replay['feasible_within_authority']
                        == w['feasible_within_authority'],
                        'authority verdict does not reproduce')
                np.testing.assert_allclose(arrays[name+'required_shift'],
                                           replay['required_shift_native'],
                                           rtol=tol, atol=tol)
                np.testing.assert_allclose(arrays[name+'t'], [w['t']], rtol=tol, atol=tol)
                within_any |= w['feasible_within_authority']
                minima.append(w['t'])
            require(len(seen) == len(cand['witnesses']), 'duplicate witness prefixes')
            require(seen == {prefix} | {wp for wp in all_witnesses
                                        if wp.startswith(prefix)},
                    'witness set does not cover every saved support array')
            measured = [w for w in cand['witnesses'] if w['status'] == 'measured']
            require(cand['minimum_t'] == (min((w['t'] for w in measured), default=None))
                    and cand['within_authority_witnesses']
                    == sum(w['feasible_within_authority'] for w in measured)
                    and cand['unresolved_witnesses']
                    == sum(w['status'] == 'inconclusive' for w in cand['witnesses']),
                    'candidate summary mismatch')
        outcome = ('inconclusive' if unresolved or not minima else
                   'within_combined_authority' if within_any else
                   'beyond_combined_authority')
        require(result['scientific_outcome'] == outcome
                and (result['minimum_t'] == (min(minima) if minima else None)),
                'recorded outcome does not match replayed margins')
        verdict.update(status='completed', evidence_valid=True,
                       scientific_outcome=outcome, witnesses_checked=checked,
                       unresolved_witnesses=unresolved,
                       minimum_t=result['minimum_t'],
                       result_sha256=file_hash(out/'result.json'),
                       observations_sha256=result['observations']['sha256'],
                       scope='Saved-input LP replay in the same SciPy/HiGHS family; '
                             'not independent physics or biological validation')
    except Exception as exc:
        verdict.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(out/'verification.json', verdict)
    return verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.workspace) if args.verify else run(args.workspace), indent=2))
