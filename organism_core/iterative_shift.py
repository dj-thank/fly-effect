"""Bounded iterate: evaluate pose -> margin LP at new geometry -> shift (FE-02).

The margin-directed physical shift (posture_shift.py) moved each passive
joint once by delta*/k and did not reach equilibrium: real geometry changes
the contact map and bias. But replaying the margin LP on the shifted poses'
saved witnesses showed the needed shift CONTRACTING at several candidates
(e.g. candidate 1: t 0.0017 -> 0.0001). This module iterates the loop a
bounded number of times per seed witness: at each iterate the unchanged
gate chain runs, and the minimum-margin root-feasible sample supplies the
next passive displacement delta*_i/k_d.

A lineage stops when a sample reaches passive-subsystem or full support,
when no root-feasible sample remains, when the driver margin exceeds the
authority box (t>1), or at the iteration cap. Each iterate's driver support
matrices, t and delta* are saved so the verifier can re-derive the whole
q sequence and replay every LP. No dynamic hold, standing, walking, CNS
execution or biological claim follows from any outcome.
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
from .passive_margin import achievable_intervals, margin_solve
from .passive_posture import (FEASIBLE_POSE, FIELDS, GATE_ORDER,
                              _evaluate_trial, passive_rows, solve_subsystem)

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PROTOCOL = {
    'id': 'FE-02-iterative-shift-v1', 'seed': 0,
    'iteration': ('evaluate gate chain -> margin LP on the minimum-t '
                  'root-feasible sample -> passive joints move by delta*/k'),
    'seeds_per_candidate': 2,
    'seeds': 'lowest-t within-authority margin witnesses per candidate',
    'max_iterations': 6,
    'driver': 'minimum-margin root-feasible sample; ties break by shallower depth',
    'stop': ['passive_subsystem_feasible', 'full_support', 'no_root_feasible_sample',
             'beyond_authority', 'iteration_cap'],
    'depths_native': [.001, .005, .01, .02, .04],
    'bracket_half_range_native': 2., 'bracket_iterations': 25,
    'root_solver_time_limit_s': 2.,
    'max_solver_calls_per_candidate': 640,
    'max_saved_driver_sets_per_candidate': 24,
    'candidate_wall_s': 120., 'worker_wall_s': 900., 'memory_mib': 8192,
    'moved_dofs': 'exactly-zero tendon-map rows except the six free-root DOFs',
    'gates': list(GATE_ORDER),
    'outcomes': ['iterated_shift_reached_full_support',
                 'iterated_shift_reached_passive_subsystem',
                 'no_convergence_in_bounded_iteration',
                 'inconclusive', 'invalid_experiment'],
    'dynamic_trials': 0, 'biological_validation': False,
    'scope': ('Bounded iterate of margin-directed physical shifts; each move '
              'is real geometry so contact map and bias are re-derived'),
}


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _check_terminal_iterate(status, irec, samples, is_last):
    """Fail-closed check that iterate records agree with the lineage's
    recorded stopping status. Non-terminal iterates must carry a
    within-authority driver (otherwise the lineage would have stopped);
    the terminal iterate must justify the recorded status."""
    from .root_ab_evidence import require
    if not is_last:
        require('driver_sample' in irec and irec['driver_margin_t'] <= 1.,
                'iterate stopped without a within-authority driver')
        return
    if status == 'reached_full_support':
        require(irec['any_full_feasible'],
                'full-support status without a feasible sample')
    if status == 'reached_passive_subsystem':
        require(irec['any_passive_feasible'],
                'passive status without a feasible sample')
    if status == 'beyond_authority':
        require(irec.get('driver_margin_t', 0.) > 1.,
                'beyond-authority lineage ended inside authority')
    if status == 'no_root_feasible_sample':
        require(all(s.get('root_balance', {}).get('feasible') is not True
                    for s in samples),
                'root-feasible sample exists at last iterate')
    if status == 'margin_unresolved':
        require(any(s.get('root_balance', {}).get('feasible') is True
                    for s in samples)
                and all(s.get('margin', {}).get('status') != 'measured'
                        for s in samples
                        if s.get('root_balance', {}).get('feasible') is True),
                'margin-unresolved lineage had a measured driver')


def _iterate(body, transmission, q0, fmax, l0, *, passive_meta, deadline,
             driver_sets):
    """Run the bounded shift iteration for one seed pose.

    passive_meta = (qpos_adrs, stiffness, ranges, passive_rows). Each iterate
    re-runs the unchanged gate chain; the minimum-margin root-feasible sample
    drives the next displacement. Returns (record, arrays).
    """
    import mujoco as mj
    from .pose_probe import first_contact_height
    from .support_diagnostics import contact_columns, evaluate_support
    from .force_audit import audit_static_forces
    from .foot_placement import root_balance, select_sample
    model, data = body.m, body.d
    qpos_adrs, stiffness, ranges, passive = passive_meta
    arrays = {}
    record = {'solver_calls': 0, 'iterates': [], 'status': 'pending'}
    q = np.asarray(q0, dtype=float).copy()
    for it in range(PROTOCOL['max_iterations']+1):
        if time.monotonic() >= deadline:
            raise TimeoutError('Iterative-shift wall budget exceeded')
        irec = {'iterate': it, 'solver_calls': 0, 'samples': [],
                'qpos': q.tolist()}
        height = first_contact_height(model, data, q, deadline=deadline)
        irec['first_contact_height_native'] = float(height)
        for n, depth in enumerate(PROTOCOL['depths_native']):
            data.qpos[:] = q; data.qpos[2] = height-depth
            data.qvel[:] = 0; data.qacc_warmstart[:] = 0
            data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
            mj.mj_forward(model, data)
            if np.any(data.warning.number):
                raise ValueError('Solver warning in iterative-shift probe')
            C, mu, contacts, nonfoot = contact_columns(model, data)
            legs = [leg for leg in LEGS
                    if any('/'+leg+'_tarsus' in c['geom'] for c in contacts)]
            sample = {'index': n, 'depth_native': depth, 'foot_legs': legs,
                      'nonfoot_contacts': nonfoot, 'contacts': contacts,
                      'root_balance': {'feasible': False, 'status': 'not_eligible'}}
            sp = f'it_{it:02d}_depth_{n:02d}_'
            arrays.update({sp+'qpos': data.qpos.copy(),
                           sp+'contact_map': C[:6].copy(),
                           sp+'target': (data.qfrc_bias-data.qfrc_passive)[:6].copy(),
                           sp+'friction': mu.copy()})
            if contacts and not nonfoot:
                sample['root_balance'] = root_balance(arrays[sp+'contact_map'],
                                                      arrays[sp+'target'], mu)
                irec['solver_calls'] += 1
            irec['samples'].append(sample)
        drivers = []
        solved = []
        for sample in irec['samples']:
            if sample['root_balance'].get('feasible') is not True:
                continue
            n = sample['index']; sp = f'it_{it:02d}_depth_{n:02d}_'
            data.qpos[:] = arrays[sp+'qpos']; data.qvel[:] = 0
            data.qacc_warmstart[:] = 0; data.qfrc_applied[:] = 0
            data.xfrc_applied[:] = 0
            mj.mj_forward(model, data)
            limits = fmax*np.exp(-((data.ten_length/l0-1)/.5)**2)
            support, inputs = evaluate_support(body, transmission, limits)
            irec['solver_calls'] += 1
            subsystem = solve_subsystem(*(inputs[k] for k in FIELDS[:4]),
                                        inputs['support_friction'],
                                        list(range(6))+passive)
            irec['solver_calls'] += 1
            intervals = achievable_intervals(stiffness, ranges,
                                             data.qpos[qpos_adrs])
            margin = margin_solve(*(inputs[k] for k in FIELDS),
                                  list(range(6)), passive, intervals)
            irec['solver_calls'] += 1
            sample['passive_rows'] = passive
            sample['passive_subsystem'] = {'status': subsystem['status'],
                                           'solver_status': subsystem['solver_status'],
                                           'subsystem_rows': subsystem['subsystem_rows']}
            sample['support'] = {'status': support['status'] if not support['feasible']
                                 else FEASIBLE_POSE,
                                 'solver_status': support['solver_status']}
            sample['margin'] = {'status': margin['status'],
                                'solver_status': margin['solver_status']}
            if margin['status'] == 'measured':
                sample['margin'].update(t=margin['t'],
                    feasible_within_authority=margin['feasible_within_authority'],
                    required_shift_native=margin['required_shift_native'])
                drivers.append((margin['t'], sample['depth_native'], n))
            audit, forces = audit_static_forces(model, data.qpos)
            solved.append((sample, inputs, intervals, audit, forces,
                           data.ten_length.copy()))
            if sample['support']['status'] == FEASIBLE_POSE:
                sample['gate_index'] = 4
            elif subsystem['status'] == 'feasible':
                sample['gate_index'] = 3
            else:
                sample['gate_index'] = 2
        driver_index = min(drivers)[2] if drivers else None
        for sample, inputs, intervals, audit, forces, ten_len in solved:
            n = sample['index']; sp = f'it_{it:02d}_depth_{n:02d}_'
            is_driver = n == driver_index
            if (driver_sets[0] < PROTOCOL['max_saved_driver_sets_per_candidate']
                    or is_driver or sample['gate_index'] >= 3):
                driver_sets[0] += 1
                if not audit['passed'] or not np.allclose(
                        forces['support_target'], inputs['support_target'],
                        rtol=1e-12, atol=1e-12):
                    raise ValueError('Iterate support force accounting mismatch')
                sample['force_accounting'] = audit
                arrays.update({sp+k: v for k, v in inputs.items()})
                arrays[sp+'tendon_lengths'] = ten_len
                arrays.update({sp+'force_'+k: v for k, v in forces.items()})
                arrays[sp+'margin_intervals'] = intervals
                sample['evidence_saved'] = True
            else:
                sample['evidence_saved'] = False
        for sample in irec['samples']:
            if 'gate_index' not in sample:
                sample['gate_index'] = (
                    1 if sample['foot_legs'] and not sample['nonfoot_contacts']
                    else 0)
        irec['best_gate'] = max(s['gate_index'] for s in irec['samples'])
        irec['any_passive_feasible'] = any(
            s.get('passive_subsystem', {}).get('status') == 'feasible'
            for s in irec['samples'])
        irec['any_full_feasible'] = any(
            s.get('support', {}).get('status') == FEASIBLE_POSE
            for s in irec['samples'])
        record['solver_calls'] += irec['solver_calls']
        record['iterates'].append(irec)
        if irec['any_full_feasible']:
            record['status'] = 'reached_full_support'
            break
        if irec['any_passive_feasible']:
            record['status'] = 'reached_passive_subsystem'
            break
        if not drivers:
            record['status'] = ('no_root_feasible_sample' if not solved
                                else 'margin_unresolved')
            break
        t_best, _, n_best = min(drivers)
        driver_sample = irec['samples'][n_best]
        irec['driver_sample'] = n_best
        irec['driver_margin_t'] = float(t_best)
        if t_best > 1.:
            record['status'] = 'beyond_authority'
            break
        delta = np.asarray(driver_sample['margin']['required_shift_native'],
                           dtype=float)
        for i, adr in enumerate(qpos_adrs):
            if stiffness[i] > 0:
                q[adr] = q[adr]+delta[i]/stiffness[i]
        if (np.any(q[qpos_adrs] < ranges[:, 0]-1e-9)
                or np.any(q[qpos_adrs] > ranges[:, 1]+1e-9)):
            record['status'] = 'left_joint_limits'
            break
    else:
        record['status'] = 'iteration_cap'
    if record['status'] == 'pending':
        record['status'] = 'iteration_cap'
    return record, arrays


def run(workspace):
    """Iterate margin-directed shifts from the best within-authority seeds."""
    import mujoco as mj
    from .body import Body
    from .root_ab import derive_xml, audit_models, file_hash, write_json
    from .root_ab_evidence import read_json
    from .transmission import SiteTendonTransmission
    workspace = Path(workspace).resolve()
    out = workspace/'iterative-shift'; out.mkdir(exist_ok=False)
    placed_dir = workspace/'foot-placement'; posture_dir = workspace/'passive-posture'
    margin_dir = workspace/'passive-margin'; source_dir = workspace/'source-study'
    started = time.monotonic(); deadline = started+PROTOCOL['worker_wall_s']
    result = {'schema': 1, 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
              'status': 'running', 'scientific_outcome': 'not_run', 'candidates': [],
              'biological_validation': False, 'walking_claimed': False,
              'standing_claimed': False, 'CNS_executed': False,
              'dynamic_trials_executed': 0, 'new_physics_trajectories': 0}
    arrays = {}
    try:
        record_path = Path(__file__).resolve().parents[1]/'examples/fe02-iterative-shift.json'
        if read_json(record_path)['parameters'] != PROTOCOL:
            raise ValueError('Versioned record differs from implemented protocol')
        margin_verdict = read_json(margin_dir/'verification.json')
        if margin_verdict.get('evidence_valid') is not True:
            raise ValueError('Passive-margin evidence was not verified')
        inputs = [placed_dir/'result.json', placed_dir/'observations.npz',
                  placed_dir/'model-receipt.json', workspace/'protocol.json',
                  source_dir/'result.json', source_dir/'observations.npz',
                  posture_dir/'result.json', posture_dir/'observations.npz',
                  margin_dir/'result.json', margin_dir/'observations.npz',
                  margin_dir/'verification.json']
        record_inputs = {p.relative_to(workspace).as_posix(): file_hash(p) for p in inputs}
        code = {name: file_hash(Path(__file__).with_name(name)) for name in
                ('iterative_shift.py', 'passive_margin.py', 'passive_posture.py',
                 'static_support.py', 'foot_placement.py', 'support_diagnostics.py',
                 'pose_probe.py', 'root_ab_evidence.py')}
        frozen = {'schema': 1, 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
                  'input_sha256': record_inputs, 'code_sha256': code,
                  'versions': {n: importlib.metadata.version(n)
                               for n in ('numpy', 'scipy', 'mujoco', 'flygym')}}
        write_json(out/'protocol.json', frozen)
        source = read_json(source_dir/'result.json')
        placed = read_json(placed_dir/'result.json')
        posture = read_json(posture_dir/'result.json')
        margin = read_json(margin_dir/'result.json')
        legacy = Body('muscle_compliance', 'tendon_candidate')
        if legacy.digest != source['body_sha256']:
            raise ValueError('Source model identity mismatch')
        xml = derive_xml(legacy.xml); model = mj.MjModel.from_xml_string(xml)
        invariants = audit_models(legacy.m, model)
        receipt = read_json(placed_dir/'model-receipt.json')
        expected = {'parent_body_sha256': legacy.digest,
                    'candidate_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
                    'compiled_invariants': invariants}
        if any(receipt[k] != v for k, v in expected.items()):
            raise ValueError('Derived model differs from the placement receipt')
        result.update(model_identity=receipt['model_identity'],
                      input_sha256=record_inputs, code_sha256=code,
                      versions=frozen['versions'], units=source['units'])
        body = Body.__new__(Body); body.m = model; body.d = mj.MjData(model)
        data = body.d; transmission = SiteTendonTransmission(model)
        with np.load(source_dir/'observations.npz', allow_pickle=False) as z:
            q0 = z['neutral_qpos'].copy()
        with np.load(placed_dir/'observations.npz', allow_pickle=False) as z:
            p_arrays = {k: z[k].copy() for k in z.files}
        with np.load(posture_dir/'observations.npz', allow_pickle=False) as z:
            t_arrays = {k: z[k].copy() for k in z.files}
        with np.load(margin_dir/'observations.npz', allow_pickle=False) as z:
            m_arrays = {k: z[k].copy() for k in z.files}
        data.qpos[:] = q0; data.qvel[:] = 0
        data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        l0 = data.ten_length.copy()
        fmax = np.asarray(source['maximum_tensions_native'], dtype=float)
        if (model.ntendon != 90 or model.nu != 0 or fmax.shape != (90,)
                or not np.isfinite(fmax).all() or np.any(fmax <= 0) or np.any(l0 <= 0)):
            raise ValueError('Original unactuated 90-tendon model and positive bounds required')
        arrays.update(neutral_qpos=q0, neutral_lengths=l0, maximum_tensions=fmax)
        margin_by_index = {c['index']: c for c in margin['candidates']}
        for entry_src in posture['candidates']:
            index = entry_src['index']; prefix = f'pose_{index:02d}_'
            candidate = placed['candidates'][index]
            entry = {'index': index, 'lineages': [], 'solver_calls': 0}
            result['candidates'].append(entry)
            if 'support' not in candidate or index not in margin_by_index:
                entry['status'] = 'no_margin_candidate'
                continue
            root_dofs = candidate['force_accounting']['root_dofs']
            passive = [int(i) for i in t_arrays[prefix+'passive_dofs']]
            qpos_adrs = t_arrays[prefix+'passive_qpos_adrs']
            ranges = t_arrays[prefix+'passive_ranges']
            stiffness = t_arrays[prefix+'passive_stiffness']
            if passive != passive_rows(p_arrays[prefix+'support_tendon_map'], root_dofs):
                raise ValueError('Passive row set differs between saved matrices')
            entry['passive_dofs'] = passive
            cand_margin = margin_by_index[index]
            declared = [w for w in cand_margin['witnesses']
                        if w['status'] == 'measured'
                        and w.get('feasible_within_authority')]
            declared.sort(key=lambda w: (w['t'], w['witness']))
            seeds = declared[:PROTOCOL['seeds_per_candidate']]
            entry['declared_lineages'] = len(seeds)
            candidate_deadline = min(deadline,
                                     time.monotonic()+PROTOCOL['candidate_wall_s'])
            driver_sets = [0]
            for w in seeds:
                if time.monotonic() >= candidate_deadline:
                    entry['lineages'].append({'meta': {'kind': 'not_run'},
                                              'status': 'budget_exhausted',
                                              'remaining_lineages_unexecuted': True})
                    break
                if entry['solver_calls'] >= PROTOCOL['max_solver_calls_per_candidate']:
                    entry['lineages'].append({'meta': {'kind': 'not_run'},
                                              'status': 'call_budget_exhausted',
                                              'remaining_lineages_unexecuted': True})
                    break
                src_arrays = p_arrays if w['source'] == 'foot-placement' else t_arrays
                q_base = src_arrays[w['prefix']+'support_qpos']
                meta = {'kind': 'seeded_iterate',
                        'source_witness': w['witness'], 'source': w['source'],
                        'source_prefix': w['prefix'], 'margin_t': w['t']}
                try:
                    record, it_arrays = _iterate(
                        body, transmission, q_base, fmax, l0,
                        passive_meta=(qpos_adrs, stiffness, ranges, passive),
                        deadline=candidate_deadline, driver_sets=driver_sets)
                except ValueError as exc:
                    if 'not bracketed' in str(exc):
                        entry['lineages'].append({'meta': meta, 'status': 'not_bracketed',
                                                  'reason': str(exc)})
                        continue
                    raise
                lp = prefix+f'lineage_{len(entry["lineages"]):02d}_'
                arrays.update({lp+k: v for k, v in it_arrays.items()})
                arrays[lp+'seed_qpos'] = np.asarray(q_base, dtype=float)
                record['meta'] = meta
                entry['lineages'].append(record)
                entry['solver_calls'] += record['solver_calls']
            status_rank = {'reached_full_support': 4, 'reached_passive_subsystem': 3,
                           'beyond_authority': 2, 'iteration_cap': 1,
                           'no_root_feasible_sample': 0, 'left_joint_limits': 0}
            done = [l['status'] for l in entry['lineages']
                    if l.get('status') != 'not_bracketed' and 'iterates' in l]
            entry.update(status='completed', lineages_executed=len(done),
                         best_status=max(done, key=lambda s: status_rank.get(s, -1),
                                         default='none'),
                         passive_reached=sum(s == 'reached_passive_subsystem'
                                             for s in done),
                         full_reached=sum(s == 'reached_full_support' for s in done))
        after = {p.relative_to(workspace).as_posix(): file_hash(p) for p in inputs}
        if after != result['input_sha256']:
            raise ValueError('Input evidence changed during the iterate run')
        n_passive = sum(c.get('passive_reached', 0) for c in result['candidates'])
        n_full = sum(c.get('full_reached', 0) for c in result['candidates'])
        unexecuted = any(l.get('remaining_lineages_unexecuted')
                         for c in result['candidates'] for l in c['lineages'])
        unresolved = any(l.get('status') == 'margin_unresolved'
                         for c in result['candidates'] for l in c['lineages'])
        if unexecuted or unresolved:
            outcome = 'inconclusive'
        elif n_full:
            outcome = 'iterated_shift_reached_full_support'
        elif n_passive:
            outcome = 'iterated_shift_reached_passive_subsystem'
        else:
            outcome = 'no_convergence_in_bounded_iteration'
        result.update(status='completed', scientific_outcome=outcome,
                      solver_calls=sum(c['solver_calls'] for c in result['candidates']),
                      lineages_executed=sum(c.get('lineages_executed', 0)
                                            for c in result['candidates']))
    except Exception as exc:
        result.update(status='inconclusive' if isinstance(exc, TimeoutError) else 'failed',
                      scientific_outcome='inconclusive' if isinstance(exc, TimeoutError)
                      else 'invalid_experiment',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if arrays:
            np.savez_compressed(out/'observations.npz', **arrays)
            result['observations'] = {'file': 'observations.npz',
                                      'sha256': file_hash(out/'observations.npz')}
        result['wall_seconds'] = time.monotonic()-started
        write_json(out/'result.json', result)
    return result


def verify(workspace):
    """Replay the iterate chain: re-derive q sequence, re-solve every saved LP."""
    from .foot_placement import (root_balance, root_contact_map,
                                 validate_root_result, validate_sample)
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import check_force, read_json, require
    workspace = Path(workspace); out = workspace/'iterative-shift'
    verdict = {'status': 'failed', 'evidence_valid': False,
               'scientific_outcome': 'invalid_experiment', 'walking_claimed': False,
               'standing_claimed': False, 'biological_validation': False}
    try:
        result = read_json(out/'result.json')
        require(all(result.get(k) is False for k in ('walking_claimed', 'standing_claimed',
                'CNS_executed', 'biological_validation'))
                and type(result.get('dynamic_trials_executed')) is int
                and result['dynamic_trials_executed'] == 0
                and type(result.get('new_physics_trajectories')) is int
                and result['new_physics_trajectories'] == 0,
                'unsupported capability claim')
        require(result.get('protocol') == PROTOCOL
                and result.get('protocol_sha256') == protocol_hash(), 'protocol mismatch')
        frozen = read_json(out/'protocol.json')
        require(frozen.get('protocol') == PROTOCOL
                and frozen.get('protocol_sha256') == protocol_hash()
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
        margin = read_json(workspace/'passive-margin/result.json')
        source = read_json(workspace/'source-study/result.json')
        require(result['versions'] == source['versions']
                and result['units'] == source['units'],
                'source environment or units changed')
        with np.load(workspace/'foot-placement/observations.npz', allow_pickle=False) as z:
            p_arrays = {k: z[k].copy() for k in z.files}
        with np.load(workspace/'passive-posture/observations.npz', allow_pickle=False) as z:
            t_arrays = {k: z[k].copy() for k in z.files}
        with np.load(workspace/'passive-margin/observations.npz', allow_pickle=False) as z:
            m_arrays = {k: z[k].copy() for k in z.files}
        margin_by_index = {c['index']: c for c in margin['candidates']}
        lineages_checked = witnesses_checked = 0
        passive_feasible = full_feasible = 0
        inconclusive = False
        for entry in result['candidates']:
            index = entry['index']; prefix = f'pose_{index:02d}_'
            candidate = placed['candidates'][index]
            require(candidate['index'] == index, 'candidate ordering changed')
            if entry.get('status') == 'no_margin_candidate':
                require('support' not in candidate or index not in margin_by_index,
                        'skipped candidate actually had margin evidence')
                continue
            require(entry['status'] == 'completed', 'candidate record incomplete')
            root_dofs = candidate['force_accounting']['root_dofs']
            passive = [int(i) for i in t_arrays[prefix+'passive_dofs']]
            qpos_adrs = t_arrays[prefix+'passive_qpos_adrs']
            ranges = t_arrays[prefix+'passive_ranges']
            stiffness = t_arrays[prefix+'passive_stiffness']
            require(passive == passive_rows(p_arrays[prefix+'support_tendon_map'],
                                            root_dofs)
                    and entry['passive_dofs'] == passive,
                    'saved passive rows are not the exact-zero rows')
            cand_margin = margin_by_index[index]
            declared = [w for w in cand_margin['witnesses']
                        if w['status'] == 'measured'
                        and w.get('feasible_within_authority')]
            declared.sort(key=lambda w: (w['t'], w['witness']))
            seeds = declared[:PROTOCOL['seeds_per_candidate']]
            require(len(entry['lineages']) <= entry['declared_lineages'] == len(seeds),
                    'declared lineage count mismatch')
            for l_i, lineage in enumerate(entry['lineages']):
                meta = lineage['meta']
                if meta['kind'] == 'not_run':
                    require(l_i == len(entry['lineages'])-1
                            and lineage.get('remaining_lineages_unexecuted') is True
                            and lineage['status'] in ('budget_exhausted',
                                                      'call_budget_exhausted'),
                            'unexecuted lineages must end the record explicitly')
                    inconclusive = True
                    continue
                require(meta['kind'] == 'seeded_iterate', 'unknown lineage kind')
                w = seeds[l_i]
                require(meta['source_witness'] == w['witness']
                        and meta['source'] == w['source']
                        and meta['source_prefix'] == w['prefix']
                        and meta['margin_t'] == w['t'], 'lineage metadata mismatch')
                src_arrays = (p_arrays if w['source'] == 'foot-placement'
                              else t_arrays)
                q_expected = src_arrays[w['prefix']+'support_qpos'].copy()
                lp = prefix+f'lineage_{l_i:02d}_'
                np.testing.assert_array_equal(arrays[lp+'seed_qpos'], q_expected)
                if lineage['status'] == 'not_bracketed':
                    require(not lineage.get('iterates')
                            and not any(k.startswith(lp) for k in arrays),
                            'unbracketed lineage has iterates or arrays')
                    continue
                require(lineage['status'] in ('reached_full_support',
                        'reached_passive_subsystem', 'no_root_feasible_sample',
                        'margin_unresolved', 'beyond_authority', 'iteration_cap',
                        'left_joint_limits'), 'unknown lineage status')
                if lineage['status'] == 'margin_unresolved':
                    inconclusive = True
                lineages_checked += 1
                for i_i, irec in enumerate(lineage['iterates']):
                    it = irec['iterate']
                    np.testing.assert_array_equal(
                        np.asarray(irec['qpos'], dtype=float), q_expected)
                    samples = irec['samples']
                    require([s['index'] for s in samples]
                            == list(range(len(PROTOCOL['depths_native'])))
                            and [s['depth_native'] for s in samples]
                            == PROTOCOL['depths_native'],
                            'incomplete fixed depth grid')
                    for sample in samples:
                        sp = lp+f'it_{it:02d}_depth_{sample["index"]:02d}_'
                        q = q_expected.copy()
                        q[2] = (irec['first_contact_height_native']
                                - sample['depth_native'])
                        np.testing.assert_array_equal(arrays[sp+'qpos'], q)
                        validate_sample(sample, arrays[sp+'contact_map'],
                                        arrays[sp+'friction'])
                        np.testing.assert_allclose(
                            arrays[sp+'contact_map'],
                            root_contact_map(q, sample['contacts']),
                            rtol=1e-12, atol=1e-12)
                        if sample['foot_legs'] and not sample['nonfoot_contacts']:
                            validate_root_result(sample['root_balance'],
                                                 arrays[sp+'contact_map'],
                                                 arrays[sp+'target'],
                                                 arrays[sp+'friction'])
                            recalculated = root_balance(
                                arrays[sp+'contact_map'], arrays[sp+'target'],
                                arrays[sp+'friction'])
                            if (recalculated['feasible']
                                    != sample['root_balance']['feasible']
                                    or recalculated['status']
                                    != sample['root_balance']['status']):
                                raise ValueError(
                                    'Iterate root balance does not reproduce')
                            if recalculated['status'] in ('invalid_solver_solution',
                                                          'residual_check_failed'):
                                raise ValueError('Invalid root solver result')
                            inconclusive |= recalculated['status'] == 'inconclusive'
                        if 'passive_subsystem' in sample or 'support' in sample:
                            require(sample['root_balance'].get('feasible') is True,
                                    'support gates ran without root balance')
                            require('passive_subsystem' in sample
                                    and 'support' in sample
                                    and 'margin' in sample, 'partial gate record')
                            rows = list(range(6))+sample['passive_rows']
                            require(sample['passive_subsystem'].get('subsystem_rows')
                                    == rows and rows[6:] == passive,
                                    'subsystem rows are not root+exact-zero rows')
                            if sample['evidence_saved'] is True:
                                witnesses_checked += 1
                                np.testing.assert_array_equal(
                                    arrays[sp+'support_qpos'], arrays[sp+'qpos'])
                                np.testing.assert_allclose(
                                    arrays[sp+'support_contact_map'][:6],
                                    arrays[sp+'contact_map'], rtol=1e-12, atol=1e-12)
                                np.testing.assert_allclose(
                                    arrays[sp+'support_target'][:6],
                                    arrays[sp+'target'], rtol=1e-12, atol=1e-12)
                                np.testing.assert_array_equal(
                                    arrays[sp+'support_friction'],
                                    arrays[sp+'friction'])
                                limits = (arrays['maximum_tensions']
                                          * np.exp(-((arrays[sp+'tendon_lengths']
                                                      / arrays['neutral_lengths']
                                                      - 1)/.5)**2))
                                np.testing.assert_allclose(
                                    arrays[sp+'support_limits'], limits,
                                    rtol=1e-12, atol=1e-12)
                                np.testing.assert_allclose(
                                    arrays[sp+'support_target'],
                                    arrays[sp+'force_support_target'],
                                    rtol=1e-12, atol=1e-12)
                                check_force(sample['force_accounting'], arrays,
                                            sp+'force_', 'unanchored_root', sp)
                                intervals = achievable_intervals(
                                    stiffness, ranges,
                                    arrays[sp+'support_qpos'][qpos_adrs])
                                np.testing.assert_allclose(
                                    arrays[sp+'margin_intervals'], intervals,
                                    rtol=0, atol=0)
                                replay = solve_subsystem(
                                    *(arrays[sp+k] for k in FIELDS[:4]),
                                    arrays[sp+'support_friction'], rows)
                                require(replay['status']
                                        == sample['passive_subsystem']['status']
                                        and replay['solver_status']
                                        == sample['passive_subsystem']
                                        ['solver_status'],
                                        'passive subsystem does not reproduce')
                                if replay['status'] in ('invalid_solver_solution',
                                                        'residual_check_failed'):
                                    raise ValueError(
                                        'Invalid passive subsystem result')
                                inconclusive |= replay['status'] == 'inconclusive'
                                full = solve_support(*(arrays[sp+k]
                                                       for k in FIELDS))
                                expected_full = (FEASIBLE_POSE if full['feasible']
                                                 else full['status'])
                                require(full['feasible']
                                        == (sample['support']['status']
                                            == FEASIBLE_POSE)
                                        and sample['support']['status']
                                        == expected_full
                                        and full['solver_status']
                                        == sample['support']['solver_status'],
                                        'full support does not reproduce')
                                if full['status'] in ('invalid_solver_solution',
                                                      'residual_check_failed'):
                                    raise ValueError('Invalid full support result')
                                inconclusive |= full['status'] == 'inconclusive'
                                margin_replay = margin_solve(
                                    *(arrays[sp+k] for k in FIELDS),
                                    list(range(6)), passive, intervals)
                                require(margin_replay['status']
                                        == sample['margin']['status']
                                        and margin_replay['solver_status']
                                        == sample['margin']['solver_status'],
                                        'margin LP does not reproduce')
                                if sample['margin']['status'] == 'measured':
                                    require(abs(margin_replay['t']
                                                - sample['margin']['t'])
                                            <= 1e-6*max(1., sample['margin']['t'])
                                            and margin_replay[
                                                'feasible_within_authority']
                                            == sample['margin'][
                                                'feasible_within_authority'],
                                            'margin t does not reproduce')
                                    np.testing.assert_allclose(
                                        sample['margin']['required_shift_native'],
                                        margin_replay['required_shift_native'],
                                        rtol=1e-6, atol=1e-6)
                                inconclusive |= (margin_replay['status']
                                                 == 'inconclusive')
                                passive_feasible += int(
                                    replay['status'] == 'feasible')
                                full_feasible += int(full['feasible'])
                            else:
                                for gate in ('passive_subsystem', 'support'):
                                    status = sample[gate]['status']
                                    require(status not in ('feasible',
                                                           FEASIBLE_POSE),
                                            'unverifiable positive gate claim')
                                    inconclusive |= status in (
                                        'inconclusive',
                                        'invalid_solver_solution',
                                        'residual_check_failed')
                    best = max(s['gate_index'] for s in samples)
                    require(irec['best_gate'] == best
                            and irec['any_passive_feasible'] == any(
                                s.get('passive_subsystem', {}).get('status')
                                == 'feasible' for s in samples)
                            and irec['any_full_feasible'] == any(
                                s.get('support', {}).get('status')
                                == FEASIBLE_POSE for s in samples),
                            'iterate summary mismatch')
                    _check_terminal_iterate(
                        lineage['status'], irec, samples,
                        i_i == len(lineage['iterates'])-1)
                    if i_i == len(lineage['iterates'])-1:
                        break
                    driver = irec['driver_sample']
                    require(samples[driver]['evidence_saved'] is True,
                            'driver sample lacks saved matrices')
                    delta = np.asarray(
                        samples[driver]['margin']['required_shift_native'],
                        dtype=float)
                    require(irec['driver_margin_t']
                            == samples[driver]['margin']['t'],
                            'driver t mismatch')
                    for i, adr in enumerate(qpos_adrs):
                        if stiffness[i] > 0:
                            q_expected[adr] = q_expected[adr]+delta[i]/stiffness[i]
            require(entry.get('solver_calls') == sum(l.get('solver_calls', 0)
                    for l in entry['lineages'])
                    and entry.get('lineages_executed') == sum(
                        'iterates' in l for l in entry['lineages'])
                    and entry.get('passive_reached') == sum(
                        l.get('status') == 'reached_passive_subsystem'
                        for l in entry['lineages'])
                    and entry.get('full_reached') == sum(
                        l.get('status') == 'reached_full_support'
                        for l in entry['lineages']), 'candidate summary mismatch')
        outcome = result['scientific_outcome']
        unresolved = any(l.get('remaining_lineages_unexecuted')
                         or l.get('status') == 'margin_unresolved'
                         for c in result['candidates']
                         for l in c.get('lineages', []))
        expected_outcome = ('inconclusive' if unresolved else
                            'iterated_shift_reached_full_support'
                            if full_feasible else
                            'iterated_shift_reached_passive_subsystem'
                            if passive_feasible else
                            'no_convergence_in_bounded_iteration')
        require(outcome == expected_outcome,
                'recorded outcome does not match replayed gates')
        verdict.update(status='completed', evidence_valid=True,
                       scientific_outcome=('inconclusive' if inconclusive
                                           or outcome == 'inconclusive'
                                           else outcome),
                       lineages_checked=lineages_checked,
                       witnesses_checked=witnesses_checked,
                       passive_feasible_samples=passive_feasible,
                       full_feasible_samples=full_feasible,
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
