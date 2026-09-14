"""Directed physical posture shifts along saved margin directions (FE-02).

The combined-authority margin (`passive_margin.py`) is a fixed-geometry
screen: it holds the contact map and bias fixed and only asks whether
achievable passive spring shifts could balance the equations. This module
performs the corresponding PHYSICAL trial — each passive joint moves to
q'_d = q_d + delta*_d/k_d from a margin witness judged within authority —
then the unchanged gate chain (foot-only contact -> root balance -> passive
subsystem -> full support) is re-evaluated at the new geometry, where the
contact map and bias really do move.

A positive gate result here is a physical static witness at a diagnostic
initial placement; it is still not a dynamic hold, standing, walking or any
biological claim. Unresolved solver states are never physical infeasibility.
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
from .passive_posture import (FEASIBLE_POSE, FIELDS, GATE_ORDER,
                              _evaluate_trial, passive_rows, solve_subsystem)

PROTOCOL = {
    'id': 'FE-02-posture-shift-v1', 'seed': 0,
    'shift': ('directed: passive DOF d moves to q_d + delta*_d/k_d from each '
              'margin witness judged within combined authority (t<=1), scale 1.0'),
    'trials': 'every margin witness with feasible_within_authority, ordered by t',
    'depths_native': [.001, .005, .01, .02, .04],
    'bracket_half_range_native': 2., 'bracket_iterations': 25,
    'root_solver_time_limit_s': 2.,
    'max_solver_calls_per_candidate': 512,
    'max_saved_support_witnesses_per_candidate': 16,
    'candidate_wall_s': 90., 'worker_wall_s': 600., 'memory_mib': 8192,
    'moved_dofs': 'exactly-zero tendon-map rows except the six free-root DOFs',
    'gates': list(GATE_ORDER),
    'outcomes': ['shift_reached_full_support', 'shift_reached_passive_subsystem',
                 'no_shift_reached_equilibrium', 'inconclusive', 'invalid_experiment'],
    'dynamic_trials': 0, 'biological_validation': False,
    'scope': ('Directed physical posture trial; the margin LP is only the aim - '
              'contact geometry and bias change with the real move'),
}


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def run(workspace):
    """Re-evaluate the gate chain at every margin-directed shifted posture."""
    import mujoco as mj
    from .body import Body
    from .root_ab import derive_xml, audit_models, file_hash, write_json
    from .root_ab_evidence import read_json
    from .transmission import SiteTendonTransmission
    workspace = Path(workspace).resolve()
    out = workspace/'posture-shift'; out.mkdir(exist_ok=False)
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
        record_path = Path(__file__).resolve().parents[1]/'examples/fe02-posture-shift.json'
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
                ('posture_shift.py', 'passive_margin.py', 'passive_posture.py',
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
            entry = {'index': index, 'trials': [], 'solver_calls': 0}
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
            entry['declared_trials'] = len(declared)
            candidate_deadline = min(deadline, time.monotonic()+PROTOCOL['candidate_wall_s'])
            witness = [0]
            for w in declared:
                if time.monotonic() >= candidate_deadline:
                    entry['trials'].append({'meta': {'kind': 'not_run'},
                                            'status': 'budget_exhausted',
                                            'remaining_trials_unexecuted': True})
                    break
                if entry['solver_calls'] >= PROTOCOL['max_solver_calls_per_candidate']:
                    entry['trials'].append({'meta': {'kind': 'not_run'},
                                            'status': 'call_budget_exhausted',
                                            'remaining_trials_unexecuted': True})
                    break
                src_arrays = p_arrays if w['source'] == 'foot-placement' else t_arrays
                q_base = src_arrays[w['prefix']+'support_qpos']
                delta = m_arrays[w['witness']+'_required_shift']
                q_trial = q_base.copy()
                for i, adr in enumerate(qpos_adrs):
                    if stiffness[i] > 0:
                        q_trial[adr] = q_base[adr]+delta[i]/stiffness[i]
                    elif abs(delta[i]) > 1e-9:
                        raise ValueError('Zero-stiffness passive row has nonzero shift')
                if (np.any(q_trial[qpos_adrs] < ranges[:, 0]-1e-9)
                        or np.any(q_trial[qpos_adrs] > ranges[:, 1]+1e-9)):
                    raise ValueError('Directed shift leaves the original joint limits')
                meta = {'kind': 'directed_shift', 'source_witness': w['witness'],
                        'source': w['source'], 'source_prefix': w['prefix'],
                        'margin_t': w['t'], 'shift_scale': 1.0}
                try:
                    record, trial_arrays = _evaluate_trial(
                        body, transmission, q_trial, fmax, l0,
                        deadline=candidate_deadline, witness=witness)
                except ValueError as exc:
                    if 'not bracketed' in str(exc):
                        entry['trials'].append({'meta': meta, 'status': 'not_bracketed',
                                                'reason': str(exc)})
                        continue
                    raise
                tp = prefix+f'shift_{len(entry["trials"]):03d}_'
                arrays.update({tp+k: v for k, v in trial_arrays.items()})
                arrays[tp+'qpos'] = np.asarray(q_trial, dtype=float)
                arrays[tp+'shift_delta'] = np.asarray(delta, dtype=float)
                record['meta'] = meta
                entry['trials'].append(record)
                entry['solver_calls'] += record['solver_calls']
            gates = [t['best_gate'] for t in entry['trials'] if 'best_gate' in t]
            entry.update(status='completed', trials_executed=len(gates),
                         best_gate=max(gates) if gates else 0,
                         passive_feasible_trials=sum(
                             t.get('any_passive_feasible') is True for t in entry['trials']),
                         full_feasible_trials=sum(
                             t.get('any_full_feasible') is True for t in entry['trials']))
        after = {p.relative_to(workspace).as_posix(): file_hash(p) for p in inputs}
        if after != result['input_sha256']:
            raise ValueError('Input evidence changed during the shift run')
        n_passive = sum(c.get('passive_feasible_trials', 0) for c in result['candidates'])
        n_full = sum(c.get('full_feasible_trials', 0) for c in result['candidates'])
        unexecuted = any(t.get('remaining_trials_unexecuted')
                         for c in result['candidates'] for t in c['trials'])
        if unexecuted:
            outcome = 'inconclusive'
        elif n_full:
            outcome = 'shift_reached_full_support'
        elif n_passive:
            outcome = 'shift_reached_passive_subsystem'
        else:
            outcome = 'no_shift_reached_equilibrium'
        result.update(status='completed', scientific_outcome=outcome,
                      solver_calls=sum(c['solver_calls'] for c in result['candidates']),
                      trials_executed=sum(c.get('trials_executed', 0)
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
    """Replay every recorded gate from saved arrays; fail closed on mismatch."""
    from .foot_placement import (root_balance, root_contact_map, select_sample,
                                 validate_root_result, validate_sample)
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import check_force, read_json, require
    workspace = Path(workspace); out = workspace/'posture-shift'
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
        require(result['versions'] == source['versions'] and result['units'] == source['units'],
                'source environment or units changed')
        with np.load(workspace/'foot-placement/observations.npz', allow_pickle=False) as z:
            p_arrays = {k: z[k].copy() for k in z.files}
        with np.load(workspace/'passive-posture/observations.npz', allow_pickle=False) as z:
            t_arrays = {k: z[k].copy() for k in z.files}
        with np.load(workspace/'passive-margin/observations.npz', allow_pickle=False) as z:
            m_arrays = {k: z[k].copy() for k in z.files}
        margin_by_index = {c['index']: c for c in margin['candidates']}
        trials_checked = witnesses_checked = 0
        passive_feasible = full_feasible = 0
        inconclusive = truncated = False
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
            require(len(entry['trials']) <= entry['declared_trials'] == len(declared),
                    'declared trial count mismatch')
            for t_i, trial in enumerate(entry['trials']):
                meta = trial['meta']
                if meta['kind'] == 'not_run':
                    require(t_i == len(entry['trials'])-1
                            and trial.get('remaining_trials_unexecuted') is True
                            and trial['status'] in ('budget_exhausted',
                                                    'call_budget_exhausted'),
                            'unexecuted trials must end the record explicitly')
                    truncated = inconclusive = True
                    continue
                require(meta['kind'] == 'directed_shift', 'unknown trial kind')
                w = declared[t_i]
                require(meta['source_witness'] == w['witness']
                        and meta['source'] == w['source']
                        and meta['source_prefix'] == w['prefix']
                        and meta['margin_t'] == w['t'], 'trial metadata mismatch')
                src_arrays = p_arrays if w['source'] == 'foot-placement' else t_arrays
                q_base = src_arrays[w['prefix']+'support_qpos']
                delta = m_arrays[w['witness']+'_required_shift']
                expected_q = q_base.copy()
                for i, adr in enumerate(qpos_adrs):
                    if stiffness[i] > 0:
                        expected_q[adr] = q_base[adr]+delta[i]/stiffness[i]
                np.testing.assert_allclose(arrays[prefix+f'shift_{t_i:03d}_qpos'],
                                           expected_q, rtol=0, atol=0)
                np.testing.assert_allclose(
                    arrays[prefix+f'shift_{t_i:03d}_shift_delta'], delta, rtol=0, atol=0)
                require(np.all(expected_q[qpos_adrs] >= ranges[:, 0]-1e-9)
                        and np.all(expected_q[qpos_adrs] <= ranges[:, 1]+1e-9),
                        'declared shift outside original joint limits')
                tp = prefix+f'shift_{t_i:03d}_'
                if trial['status'] == 'not_bracketed':
                    require(not trial.get('samples')
                            and not any(k.startswith(tp) for k in arrays),
                            'unbracketed trial has samples or arrays')
                    continue
                samples = trial['samples']
                require([s['index'] for s in samples] == list(range(len(PROTOCOL['depths_native'])))
                        and [s['depth_native'] for s in samples] == PROTOCOL['depths_native'],
                        'incomplete fixed depth grid')
                trials_checked += 1
                for sample in samples:
                    sp = tp+f'depth_{sample["index"]:02d}_'
                    q = expected_q.copy()
                    q[2] = trial['first_contact_height_native']-sample['depth_native']
                    np.testing.assert_array_equal(arrays[sp+'qpos'], q)
                    validate_sample(sample, arrays[sp+'contact_map'], arrays[sp+'friction'])
                    np.testing.assert_allclose(arrays[sp+'contact_map'],
                                               root_contact_map(q, sample['contacts']),
                                               rtol=1e-12, atol=1e-12)
                    if sample['foot_legs'] and not sample['nonfoot_contacts']:
                        validate_root_result(sample['root_balance'],
                                             arrays[sp+'contact_map'],
                                             arrays[sp+'target'], arrays[sp+'friction'])
                        recalculated = root_balance(arrays[sp+'contact_map'],
                                                    arrays[sp+'target'],
                                                    arrays[sp+'friction'])
                        if (recalculated['feasible'] != sample['root_balance']['feasible']
                                or recalculated['status'] != sample['root_balance']['status']):
                            raise ValueError('Trial root balance does not reproduce')
                        if recalculated['status'] in ('invalid_solver_solution',
                                                      'residual_check_failed'):
                            raise ValueError('Invalid root solver result')
                        inconclusive |= recalculated['status'] == 'inconclusive'
                    if 'passive_subsystem' in sample or 'support' in sample:
                        require(sample['root_balance'].get('feasible') is True,
                                'support gates ran without root balance')
                        require('passive_subsystem' in sample and 'support' in sample,
                                'partial gate record')
                        rows = list(range(6))+sample['passive_rows']
                        require(sample['passive_subsystem'].get('subsystem_rows') == rows
                                and rows[6:] == passive,
                                'subsystem rows are not root+exact-zero rows')
                        if sample['evidence_saved'] is True:
                            witnesses_checked += 1
                            np.testing.assert_array_equal(arrays[sp+'support_qpos'],
                                                          arrays[sp+'qpos'])
                            np.testing.assert_allclose(arrays[sp+'support_contact_map'][:6],
                                                       arrays[sp+'contact_map'],
                                                       rtol=1e-12, atol=1e-12)
                            np.testing.assert_allclose(arrays[sp+'support_target'][:6],
                                                       arrays[sp+'target'],
                                                       rtol=1e-12, atol=1e-12)
                            np.testing.assert_array_equal(arrays[sp+'support_friction'],
                                                          arrays[sp+'friction'])
                            limits = (arrays['maximum_tensions']
                                      * np.exp(-((arrays[sp+'tendon_lengths']
                                                  / arrays['neutral_lengths']-1)/.5)**2))
                            np.testing.assert_allclose(arrays[sp+'support_limits'], limits,
                                                       rtol=1e-12, atol=1e-12)
                            np.testing.assert_allclose(arrays[sp+'support_target'],
                                                       arrays[sp+'force_support_target'],
                                                       rtol=1e-12, atol=1e-12)
                            check_force(sample['force_accounting'], arrays, sp+'force_',
                                        'unanchored_root', sp)
                            replay = solve_subsystem(*(arrays[sp+k] for k in FIELDS[:4]),
                                                     arrays[sp+'support_friction'], rows)
                            require(replay['status'] == sample['passive_subsystem']['status']
                                    and replay['solver_status']
                                    == sample['passive_subsystem']['solver_status'],
                                    'passive subsystem does not reproduce')
                            if replay['status'] in ('invalid_solver_solution',
                                                    'residual_check_failed'):
                                raise ValueError('Invalid passive subsystem result')
                            inconclusive |= replay['status'] == 'inconclusive'
                            full = solve_support(*(arrays[sp+k] for k in FIELDS))
                            expected_full = FEASIBLE_POSE if full['feasible'] else full['status']
                            require(full['feasible'] == (sample['support']['status'] == FEASIBLE_POSE)
                                    and sample['support']['status'] == expected_full
                                    and full['solver_status'] == sample['support']['solver_status'],
                                    'full support does not reproduce')
                            if full['status'] in ('invalid_solver_solution',
                                                  'residual_check_failed'):
                                raise ValueError('Invalid full support result')
                            inconclusive |= full['status'] == 'inconclusive'
                            passive_feasible += int(replay['status'] == 'feasible')
                            full_feasible += int(full['feasible'])
                        else:
                            for gate in ('passive_subsystem', 'support'):
                                status = sample[gate]['status']
                                require(status not in ('feasible', FEASIBLE_POSE),
                                        'unverifiable positive gate claim')
                                inconclusive |= status in ('inconclusive',
                                    'invalid_solver_solution', 'residual_check_failed')
                selected = select_sample(samples)
                expected_index = None if selected is None else selected['index']
                require(trial.get('selected_sample_index') == expected_index,
                        'selected depth differs from frozen policy')
                recomputed_best = 0
                for sample in samples:
                    gate = 0
                    if sample['foot_legs'] and not sample['nonfoot_contacts']:
                        gate = 1
                        if sample['root_balance'].get('feasible') is True:
                            gate = 2
                            if sample.get('passive_subsystem', {}).get('status') == 'feasible':
                                gate = 3
                                if sample.get('support', {}).get('status') == FEASIBLE_POSE:
                                    gate = 4
                    require(sample['gate_index'] == gate, 'per-sample gate index mismatch')
                    recomputed_best = max(recomputed_best, gate)
                require(trial['best_gate'] == recomputed_best
                        and trial['best_gate_name'] == (GATE_ORDER[recomputed_best-1]
                                                        if recomputed_best
                                                        else 'no_foot_only_contact'),
                        'trial best gate mismatch')
                require(trial['any_passive_feasible'] == any(
                    s.get('passive_subsystem', {}).get('status') == 'feasible'
                    for s in samples)
                    and trial['any_full_feasible'] == any(
                        s.get('support', {}).get('status') == FEASIBLE_POSE
                        for s in samples), 'trial feasibility flags mismatch')
            require(entry.get('solver_calls') == sum(t.get('solver_calls', 0)
                    for t in entry['trials'])
                    and entry.get('trials_executed') == sum('best_gate' in t
                    for t in entry['trials'])
                    and entry.get('passive_feasible_trials') == sum(
                        t.get('any_passive_feasible') is True for t in entry['trials'])
                    and entry.get('full_feasible_trials') == sum(
                        t.get('any_full_feasible') is True for t in entry['trials'])
                    and entry.get('best_gate') == max(
                        [t['best_gate'] for t in entry['trials'] if 'best_gate' in t],
                        default=0), 'candidate summary mismatch')
        outcome = result['scientific_outcome']
        expected_outcome = ('inconclusive' if truncated else
                            'shift_reached_full_support' if full_feasible else
                            'shift_reached_passive_subsystem' if passive_feasible else
                            'no_shift_reached_equilibrium')
        require(outcome == expected_outcome,
                'recorded outcome does not match replayed gates')
        verdict.update(status='completed', evidence_valid=True,
                       scientific_outcome='inconclusive' if inconclusive else outcome,
                       recorded_outcome=outcome, trials_checked=trials_checked,
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
