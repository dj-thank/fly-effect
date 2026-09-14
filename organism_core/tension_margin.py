"""Tension-authority margin on every retained support witness (FE-02).

The iterative shift contracted the passive-subsystem residual to solver
resolution but the full-support LP stayed infeasible at a ~0.2 scaled
residual with every tendon saturated at its declared limit. This module
measures the missing lever directly: keeping every equality row strict and
the friction diamond unchanged, scale only the tension bounds by a single
scalar s >= 0 and minimize s. The optimum s* is the muscle-force multiple
this pose and contact set would need for static support; infeasibility at
any s means contact geometry alone cannot cover the root+passive rows.

Fixed-geometry algebraic replay on saved witnesses only — no MuJoCo
evaluation, no physical posture claim, no dynamic hold, standing,
walking, CNS execution or biological validation.
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
from .passive_posture import FIELDS

WITNESS = re.compile(r'^(pose_\d+_.*)support_tendon_map$')

PROTOCOL = {
    'id': 'FE-02-tension-margin-v1',
    'question': ('With all equality rows strict and friction unchanged, what '
                 'uniform scale s on declared tension limits admits static '
                 'support at each saved witness?'),
    'variables': ('tensions f, contact forces r, scalar tension-authority '
                  'scale s; constraints f_j <= s*fmax_j, f >= 0, friction '
                  'diamond, T f + C r = b on every row'),
    'sources': ['foot-placement', 'passive-posture', 'posture-shift',
                'iterative-shift'],
    'outcomes': ['within_tension_authority', 'bounded_tension_deficit',
                 'unbounded_tension_deficit', 'inconclusive',
                 'invalid_experiment'],
    'time_limit_s': 10., 'max_solver_calls': 512, 'wall_seconds': 240.,
    'replay_tolerance': 1e-6,
    'dynamic_trials': 0, 'biological_validation': False,
    'scope': ('Fixed-geometry algebraic screen on saved witnesses; a finite '
              's* measures the force deficit of the transferred tension '
              'limits, not a physical posture or a biological claim'),
}


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def tension_margin_solve(tendon_map, contact_map, target, limits, friction):
    """Min s >= 0 s.t. f_j <= s*limit_j admits T f + C r = b on all rows.

    Same unknowns and cone constraints as the unchanged support LP plus the
    scalar s replacing each tension bound. Status 2 means no finite tension
    scale suffices — contact geometry and sign cannot cover the rows that
    tendons cannot reach (root and passive DOFs) at this witness.
    """
    from scipy.optimize import linprog
    T, C = np.asarray(tendon_map, dtype=float), np.asarray(contact_map, dtype=float)
    b, limit, mu = (np.asarray(x, dtype=float) for x in (target, limits, friction))
    if (T.ndim != 2 or C.ndim != 2 or T.shape[0] != C.shape[0] or not T.shape[0]
            or not T.shape[1] or C.shape[1] % 3 or not C.shape[1]
            or b.shape != (T.shape[0],) or limit.shape != (T.shape[1],)
            or mu.shape != (C.shape[1]//3,)):
        raise ValueError('Incompatible tension-margin dimensions')
    if (not all(np.isfinite(x).all() for x in (T, C, b, limit, mu))
            or np.any(limit <= 0) or np.any(mu < 0)):
        raise ValueError('Finite data, positive tension limits and nonnegative friction required')
    n = len(limit); k = len(mu); m = n+3*k+1
    scale = np.maximum(np.abs(b), np.max(np.abs(T)*limit, axis=1))
    scale = np.maximum(scale, 1e-6)
    A_eq = np.column_stack((T, C, np.zeros(len(b))))/scale[:, None]
    b_eq = b/scale
    ineq = np.zeros((4*k+n, m))
    for j in range(k):
        for r, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            ineq[4*j+r, n+3*j:n+3*j+3] = [sx, sy, -mu[j]]
    for j in range(n):
        ineq[4*k+j, j] = 1.
        ineq[4*k+j, -1] = -limit[j]
    b_ub = np.zeros(4*k+n)
    bounds = [(0., None)]*n + [(None, None), (None, None), (0., None)]*k
    bounds.append((0., None))
    cost = np.zeros(m); cost[-1] = 1.
    result = linprog(cost, A_ub=ineq, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs',
                     options={'time_limit': PROTOCOL['time_limit_s']})
    report = {'kind': 'tension_authority_margin', 'solver_status': int(result.status),
              'solver_message': str(result.message), 'solver_success': bool(result.success),
              'status': 'inconclusive', 'biological_validation': False,
              'scope': 'Fixed-geometry tension-authority screen on one saved '
                       'witness; not a physical posture and not a muscle model.'}
    if result.success:
        x = np.asarray(result.x, dtype=float)
        if x.shape != (m,) or not np.isfinite(x).all():
            report['status'] = 'invalid_solver_solution'
            return report
        forces = x[n:n+3*k].reshape(k, 3)
        residual = np.column_stack((T, C))@x[:n+3*k] - b
        violations = [float(np.max(np.abs(residual)/scale)),
                      float(max(0., np.max(-x[:n]))),
                      float(max(0., np.max(x[:n]/limit-x[-1]))),
                      float(max(0., np.max(-forces[:, 2]))),
                      float(max(0., np.max(ineq@x-b_ub))),
                      0. if x[-1] >= 0. else float(-x[-1])]
        frac = x[:n]/limit
        report.update(s=float(x[-1]),
                      tensions_native=x[:n].tolist(),
                      contact_forces_native=forces.tolist(),
                      equality_residual_native=residual.tolist(),
                      row_scales_native=scale.tolist(),
                      tension_fraction_of_declared=frac.tolist(),
                      saturated_tendons=int(np.sum(frac >= x[-1]-1e-9*x[-1]-1e-9)),
                      maximum_scaled_residual=violations[0],
                      maximum_constraint_violation=max(violations[1:]),
                      within_authority=bool(x[-1] <= 1.+1e-9),
                      solver_checks_passed=all(v <= 1e-6 for v in violations))
        report['status'] = 'measured' if report['solver_checks_passed'] else 'residual_check_failed'
    elif result.status == 2:
        report['status'] = 'unreachable_at_any_scale'
    return report


def _witness_sets(arrays):
    """All saved support-witness prefixes present in an observations mapping."""
    out = []
    for key in arrays:
        match = WITNESS.match(key)
        if match and all(match.group(1)+f in arrays for f in FIELDS):
            out.append(match.group(1))
    return sorted(set(out))


def _sources(workspace):
    """Declared evidence sources -> (result record, observations mapping)."""
    from .root_ab_evidence import read_json
    out = {}
    for name in PROTOCOL['sources']:
        d = workspace/name
        if not d.is_dir():
            raise FileNotFoundError(f'tension-margin requires retained {name} evidence')
        verdict = read_json(d/'verification.json')
        if verdict.get('evidence_valid') is not True:
            raise ValueError(f'{name} evidence was not verified')
        with np.load(d/'observations.npz', allow_pickle=False) as z:
            out[name] = {'result': read_json(d/'result.json'),
                         'arrays': {k: z[k].copy() for k in z.files}}
    return out


def _linkage(name, result):
    """Map a saved witness prefix -> the record that must mark it saved."""
    del result
    if name == 'foot-placement':
        def link(prefix, entry):
            pose = int(prefix.split('_')[1])
            return pose == entry['index'] and 'support' in entry
        return link
    if name == 'passive-posture':
        def link(prefix, entry):
            meta = prefix[8:].split('_')
            trial = entry['trials'][int(meta[1])]
            sample = trial['samples'][int(meta[3])]
            return (sample['index'] == int(meta[3])
                    and sample['evidence_saved'] is True)
        return link
    if name == 'posture-shift':
        def link(prefix, entry):
            meta = prefix[8:].split('_')
            trial = entry['trials'][int(meta[1])]
            sample = trial['samples'][int(meta[3])]
            return (sample['index'] == int(meta[3])
                    and sample['evidence_saved'] is True)
        return link
    if name == 'iterative-shift':
        def link(prefix, entry):
            meta = prefix[8:].split('_')
            lineage = entry['lineages'][int(meta[1])]
            iterate = lineage['iterates'][int(meta[3])]
            sample = iterate['samples'][int(meta[5])]
            return (iterate['iterate'] == int(meta[3])
                    and sample['index'] == int(meta[5])
                    and sample['evidence_saved'] is True)
        return link
    raise ValueError('unknown evidence source '+name)


def run(workspace):
    """Measure the tension-authority scale s* on every retained witness."""
    from .root_ab import file_hash, write_json
    workspace = Path(workspace).resolve()
    out = workspace/'tension-margin'; out.mkdir(exist_ok=False)
    started = time.monotonic(); deadline = started+PROTOCOL['wall_seconds']
    record = {'schema': 1, 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
              'status': 'running', 'candidates': [], 'biological_validation': False,
              'walking_claimed': False, 'standing_claimed': False,
              'CNS_executed': False, 'dynamic_trials_executed': 0,
              'new_physics_trajectories': 0}
    arrays = {}
    try:
        sources = _sources(workspace)
        inputs = []
        for name in PROTOCOL['sources']:
            d = workspace/name
            inputs += [d/'result.json', d/'observations.npz',
                       d/'verification.json']
        inputs += [workspace/'protocol.json', workspace/'source-study/result.json',
                   workspace/'source-study/observations.npz']
        record['input_sha256'] = {p.relative_to(workspace).as_posix(): file_hash(p)
                                  for p in inputs}
        record['code_sha256'] = {name: file_hash(Path(__file__).with_name(name)) for name in
                                 ('tension_margin.py', 'static_support.py',
                                  'passive_posture.py', 'root_ab_evidence.py')}
        record['versions'] = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy')}
        write_json(out/'protocol.json', record)
        placed = sources['foot-placement']['result']
        calls = 0
        for entry in placed['candidates']:
            index = entry['index']
            if 'support' not in entry:
                continue
            prefix = f'pose_{index:02d}_'
            witnesses = [{'source': 'foot-placement', 'prefix': prefix,
                          'entry': entry}]
            for name in ('passive-posture', 'posture-shift', 'iterative-shift'):
                src = sources[name]
                cand = next((c for c in src['result']['candidates']
                             if c['index'] == index), None)
                for wp in _witness_sets(src['arrays']):
                    if wp.startswith(prefix):
                        witnesses.append({'source': name, 'prefix': wp,
                                          'entry': cand, 'arrays': src['arrays']})
            cand = {'index': index, 'witnesses': [], 'solver_calls': 0,
                    'status': 'completed'}
            for w in witnesses:
                if calls >= PROTOCOL['max_solver_calls']:
                    raise TimeoutError('Tension-margin solver-call budget exceeded')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Tension-margin wall budget exceeded')
                a = (sources['foot-placement']['arrays'] if w['source'] == 'foot-placement'
                     else w['arrays'])
                wp = w['prefix']
                require_link = _linkage(w['source'], sources[w['source']]['result'])
                if w['entry'] is None or not require_link(wp, w['entry']):
                    raise ValueError('saved witness lacks evidence_saved linkage: '+wp)
                report = tension_margin_solve(a[wp+'support_tendon_map'],
                                              a[wp+'support_contact_map'],
                                              a[wp+'support_target'],
                                              a[wp+'support_limits'],
                                              a[wp+'support_friction'])
                calls += 1; cand['solver_calls'] += 1
                name = f'w_{len(arrays):04d}_'
                if report['status'] == 'measured':
                    arrays[name+'s'] = np.asarray([report['s']])
                    arrays[name+'tensions'] = np.asarray(
                        report['tensions_native'], dtype=float)
                keep = {'witness': name[:-1], 'source': w['source'], 'prefix': wp}
                keep.update({k: report[k] for k in
                             ('status', 'solver_status', 'solver_message', 'scope')
                             if k in report})
                if report['status'] == 'measured':
                    keep.update(s=report['s'],
                                within_authority=report['within_authority'],
                                saturated_tendons=report['saturated_tendons'],
                                maximum_scaled_residual=report['maximum_scaled_residual'],
                                maximum_constraint_violation=report['maximum_constraint_violation'])
                cand['witnesses'].append(keep)
            measured = [w for w in cand['witnesses'] if w['status'] == 'measured']
            cand['minimum_s'] = min((w['s'] for w in measured), default=None)
            cand['unresolved_witnesses'] = sum(
                w['status'] == 'inconclusive' for w in cand['witnesses'])
            record['candidates'].append(cand)
        after = {p.relative_to(workspace).as_posix(): file_hash(p) for p in inputs}
        if after != record['input_sha256']:
            raise ValueError('Physical source evidence changed during the margin run')
        all_w = [w for c in record['candidates'] for w in c['witnesses']]
        measured_all = [w for w in all_w if w['status'] == 'measured']
        unbounded = any(w['status'] == 'unreachable_at_any_scale' for w in all_w)
        undecided = any(w['status'] not in ('measured', 'unreachable_at_any_scale')
                        for w in all_w) or not all_w
        if undecided:
            outcome = 'inconclusive'
        elif any(w.get('within_authority') for w in measured_all):
            outcome = 'within_tension_authority'
        elif unbounded:
            outcome = 'unbounded_tension_deficit'
        else:
            outcome = 'bounded_tension_deficit'
        record.update(status='completed', scientific_outcome=outcome,
                      solver_calls=calls,
                      witnesses_checked=len(measured_all)
                      + sum(1 for w in all_w
                            if w['status'] == 'unreachable_at_any_scale')
                      + sum(c['unresolved_witnesses'] for c in record['candidates']),
                      minimum_s=min((w['s'] for w in measured_all), default=None))
    except Exception as exc:
        record.update(status='inconclusive' if isinstance(exc, TimeoutError) else 'failed',
                      scientific_outcome='inconclusive' if isinstance(exc, TimeoutError)
                      else 'invalid_experiment',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        np.savez_compressed(out/'observations.npz', **arrays)
        record['observations'] = {'file': 'observations.npz',
                                  'sha256': file_hash(out/'observations.npz')}
        record['wall_seconds'] = time.monotonic()-started
        write_json(out/'result.json', record)
    return record


def verify(workspace):
    """Replay every saved tension-margin LP from retained arrays; fail closed."""
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import read_json, require
    workspace = Path(workspace); out = workspace/'tension-margin'
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
        sources = _sources(workspace)
        checked = unresolved = 0
        within_any = False
        unbounded = False
        minima = []
        for cand in result['candidates']:
            index = cand['index']; prefix = f'pose_{index:02d}_'
            seen = set()
            for w in cand['witnesses']:
                name = w['witness']+'_'
                src = sources[w['source']]
                a = src['arrays']
                wp = w['prefix']; seen.add(wp)
                require(wp+'support_tendon_map' in a,
                        'witness prefix missing from declared source')
                entry = next((c for c in src['result']['candidates']
                              if c['index'] == index), None)
                if w['source'] == 'foot-placement':
                    entry = src['result']['candidates'][index]
                require(_linkage(w['source'], src['result'])(wp, entry),
                        'witness lacks evidence_saved linkage in '+w['source'])
                if w['status'] == 'measured':
                    require(name+'s' in arrays and name+'tensions' in arrays,
                            'missing saved margin arrays')
                replay = tension_margin_solve(a[wp+'support_tendon_map'],
                                              a[wp+'support_contact_map'],
                                              a[wp+'support_target'],
                                              a[wp+'support_limits'],
                                              a[wp+'support_friction'])
                require(replay['status'] == w['status']
                        and replay['solver_status'] == w['solver_status'],
                        'tension-margin solve does not reproduce')
                if w['status'] != 'measured':
                    unresolved += int(w['status'] != 'unreachable_at_any_scale')
                    unbounded |= w['status'] == 'unreachable_at_any_scale'
                    continue
                checked += 1
                tol = PROTOCOL['replay_tolerance']
                require(abs(replay['s']-w['s']) <= tol*max(1., w['s']),
                        'tension scale s does not reproduce')
                require(replay['within_authority'] == w['within_authority'],
                        'authority verdict does not reproduce')
                np.testing.assert_allclose(arrays[name+'s'], [w['s']],
                                           rtol=tol, atol=tol)
                np.testing.assert_allclose(arrays[name+'tensions'],
                                           replay['tensions_native'],
                                           rtol=tol, atol=tol)
                within_any |= w['within_authority']
                minima.append(w['s'])
            require(len(seen) == len(cand['witnesses']), 'duplicate witness prefixes')
            expected = {prefix}
            for name in ('passive-posture', 'posture-shift', 'iterative-shift'):
                expected |= {wp for wp in _witness_sets(sources[name]['arrays'])
                             if wp.startswith(prefix)}
            require(seen == expected,
                    'witness set does not cover every saved support array')
            measured = [w for w in cand['witnesses'] if w['status'] == 'measured']
            require(cand['minimum_s'] == (min((w['s'] for w in measured), default=None))
                    and cand['unresolved_witnesses']
                    == sum(w['status'] == 'inconclusive' for w in cand['witnesses']),
                    'candidate summary mismatch')
        outcome = ('inconclusive' if unresolved or (not minima and not unbounded) else
                   'within_tension_authority' if within_any else
                   'unbounded_tension_deficit' if unbounded else
                   'bounded_tension_deficit')
        require(result['scientific_outcome'] == outcome
                and (result['minimum_s'] == (min(minima) if minima else None)),
                'recorded outcome does not match replayed margins')
        verdict.update(status='completed', evidence_valid=True,
                       scientific_outcome=outcome, witnesses_checked=checked,
                       unresolved_witnesses=unresolved,
                       minimum_s=result['minimum_s'],
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
