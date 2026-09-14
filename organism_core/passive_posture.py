"""Bounded passive-joint posture search for root+passive equilibrium (FE-02).

Only DOFs whose tendon-map row is exactly zero (the passive tarsus chain in the
transferred muscle model) may move, and only inside the ORIGINAL joint ranges.
The free root keeps each saved candidate's XY position and orientation; every
actuated joint keeps its saved value. Each declared trial re-brackets first
ground contact, samples the unchanged five-depth grid, and is judged in the
fixed order: foot-only contact -> root balance -> passive rows -> all tendons.

This is a diagnostic initial-placement search. No dynamic hold, standing,
walking, CNS execution or biological equivalence is claimed by any outcome.
A timeout, budget exhaustion or unresolved solver result is never converted
into a physical infeasibility claim.
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

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
INFEASIBLE = 'infeasible_under_declared_constraints'
FEASIBLE_POSE = 'feasible_at_tested_pose'
GATE_ORDER = ('foot_only_contact', 'root_balance', 'passive_subsystem', 'full_support')
FIELDS = ('support_tendon_map', 'support_contact_map', 'support_target',
          'support_limits', 'support_friction')
PROTOCOL = {
    'id': 'FE-02-passive-posture-v2', 'seed': 0,
    'dof_grid_points': 5, 'seeded_box_samples': 8, 'composed_trials': 1,
    'depths_native': [.001, .005, .01, .02, .04],
    'joint_limit_inset_rad': 1e-4,
    'bracket_half_range_native': 2., 'bracket_iterations': 25,
    'root_solver_time_limit_s': 2., 'spring_shift_points': 5,
    # 160 declared trials/candidate at <=15 solver calls each; a smaller cap
    # guaranteed truncation, which the verifier correctly reports inconclusive.
    'max_solver_calls_per_candidate': 2400,
    'screen_solver_calls_per_candidate': 160,
    'max_saved_support_witnesses_per_candidate': 16,
    'candidate_wall_s': 120., 'worker_wall_s': 1200., 'memory_mib': 8192,
    'search_dofs': 'exactly-zero tendon-map rows except the six free-root DOFs',
    'control': 'unchanged saved candidate pose re-derived as the first trial',
    'gates': list(GATE_ORDER),
    'dynamic_trials': 0, 'biological_validation': False,
    'amended_from': ('FE-02-passive-posture-v1 pre-registration sized the call '
                     'budget (640) below the declared trial list; this version '
                     'only corrects that bound before any result was claimed'),
}


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _deadline(deadline):
    if not np.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Passive-posture wall budget exceeded')


def _positive(value, name, *, integer=False):
    if ((type(value) is not int if integer else type(value) not in (int, float))
            or value <= 0 or not np.isfinite(float(value))):
        raise ValueError('Positive finite ' + name + ' required')


def passive_rows(tendon_map, root_rows):
    """DOF rows with an exactly-zero tendon row, excluding declared root rows."""
    T = np.asarray(tendon_map, dtype=float)
    if T.ndim != 2 or not T.size or not np.isfinite(T).all():
        raise ValueError('Finite nonempty two-dimensional tendon map required')
    roots = []
    for i in root_rows:
        if type(i) is not int or i < 0 or i >= T.shape[0] or i in roots:
            raise ValueError('Unique in-range integer root rows required')
        roots.append(i)
    return [int(i) for i in np.flatnonzero(np.all(T == 0, axis=1)) if i not in roots]


def _grid_values(lower, upper, count):
    lo, hi = lower+PROTOCOL['joint_limit_inset_rad'], upper-PROTOCOL['joint_limit_inset_rad']
    if not lo < hi:
        raise ValueError('Empty passive joint interval')
    return np.linspace(lo, hi, count)


def trial_postures(candidate_qpos, passive_qpos_adrs, passive_ranges, candidate_index,
                   *, protocol=PROTOCOL):
    """Deterministic declared trial list; never adapts to outcomes.

    Order: unchanged control; one-DOF sweeps interleaved grid-point-major so an
    early budget stop still covers every DOF coarsely; then seeded uniform
    samples over the whole passive box. The composed trial is appended by the
    caller because its content is fixed by recorded residuals, not this list.
    """
    q0 = np.asarray(candidate_qpos, dtype=float)
    adrs = np.asarray(passive_qpos_adrs, dtype=int)
    ranges = np.asarray(passive_ranges, dtype=float)
    if (q0.ndim != 1 or not q0.size or adrs.ndim != 1 or not adrs.size
            or ranges.shape != (len(adrs), 2) or not np.isfinite(q0).all()
            or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1])):
        raise ValueError('Invalid passive search inputs')
    if (len(set(adrs.tolist())) != len(adrs) or np.any(adrs < 7) or np.any(adrs >= len(q0))
            or np.any(q0[adrs] < ranges[:, 0]) or np.any(q0[adrs] > ranges[:, 1])):
        raise ValueError('Passive addresses must be unique non-root qpos indices inside limits')
    trials = [(q0.copy(), {'kind': 'control'})]
    seen = {q0.tobytes()}
    points = int(protocol['dof_grid_points'])
    for g in range(points):
        for j, adr in enumerate(adrs):
            value = _grid_values(*ranges[j], points)[g]
            q = q0.copy(); q[adr] = value
            if q.tobytes() in seen:
                continue
            seen.add(q.tobytes())
            trials.append((q, {'kind': 'sweep', 'qpos_adr': int(adr),
                               'grid_point': int(g), 'value': float(value)}))
    inset = protocol['joint_limit_inset_rad']
    rng = np.random.RandomState([int(protocol['seed']), int(candidate_index)])
    for s in range(int(protocol['seeded_box_samples'])):
        q = q0.copy()
        q[adrs] = rng.uniform(ranges[:, 0]+inset, ranges[:, 1]-inset)
        if q.tobytes() in seen:
            continue
        seen.add(q.tobytes())
        trials.append((q, {'kind': 'seeded', 'sample': int(s)}))
    return trials


def _trial_passive_score(trial):
    """Best recorded passive-subsystem residual for one trial; None if it never ran."""
    best = None
    for sample in trial.get('samples', []):
        report = sample.get('passive_subsystem')
        if report is None:
            continue
        if report.get('status') == 'feasible':
            score = 0.
        else:
            score = report.get('nearest_balance', {}).get('minimum_scaled_balance_error')
        if score is None or not np.isfinite(score):
            continue
        if best is None or score < best:
            best = float(score)
    return best


def compose_trial(candidate_qpos, trials):
    """One extra posture from each swept DOF's best recorded subsystem residual.

    Sweep trials that never reached the passive gate contribute no score; those
    DOFs keep the saved candidate value. This is a bounded heuristic trial,
    never an optimum claim. Regenerated identically by the verifier.
    """
    q = np.asarray(candidate_qpos, dtype=float).copy()
    best = {}
    for trial in trials:
        meta = trial.get('meta', {})
        if meta.get('kind') != 'sweep':
            continue
        score = _trial_passive_score(trial)
        adr = meta['qpos_adr']
        if score is not None and (adr not in best or score < best[adr][0]):
            best[adr] = (score, float(meta['value']))
    for adr, (_, value) in best.items():
        q[adr] = value
    return q, {'kind': 'composed', 'scored_dofs': sorted(best)}


def solve_subsystem(tendon_map, contact_map, target, limits, friction, rows):
    """Original support LP restricted to declared rows; same solver/limits."""
    ids = np.asarray(rows, dtype=int)
    report = solve_support(np.asarray(tendon_map, dtype=float)[ids],
                           np.asarray(contact_map, dtype=float)[ids],
                           np.asarray(target, dtype=float)[ids], limits, friction)
    report['subsystem_rows'] = [int(i) for i in rows]
    report['scope'] = 'Necessary subsystem feasibility; not full actuated support'
    return report


def _reduced(report):
    """Keep a nonfeasible report's verdict without large force/residual vectors."""
    if report is None or report.get('status') in ('feasible', FEASIBLE_POSE):
        return report
    dropped = {'tensions_native', 'contact_forces_native', 'equality_residual_native',
               'row_scales_native'}
    keep = {k: v for k, v in report.items() if k not in dropped}
    if 'nearest_balance' in keep:
        nb = keep['nearest_balance']
        keep['nearest_balance'] = {k: nb[k] for k in
                                   ('is_feasible_solution', 'minimum_scaled_balance_error')
                                   if k in nb}
    return keep


def _spring_screen(tendon_map, contact_map, target, limits, friction, *,
                   root_rows, passive, support_qpos, qpos_adrs, stiffness, springref,
                   ranges, wall_seconds, max_calls):
    """Fixed-geometry spring-authority screen on one saved pose; NOT physical.

    For each passive row, the achievable spring term k*(q-ref) is swept inside
    the original joint interval while the contact map, bias and every other row
    stay fixed. A positive point means the row's required contact torque can
    move inside its spring range at this frozen geometry. Moving a joint really
    also moves the contact point and bias, so this screen alone never proves or
    disproves physical equilibrium; physical trials do that separately.
    """
    _positive(wall_seconds, 'wall_seconds')
    _positive(max_calls, 'max_calls', integer=True)
    deadline = time.monotonic()+wall_seconds
    q_support = np.asarray(support_qpos, dtype=float)
    rows = list(root_rows)+list(passive)
    result = {'kind': 'fixed_geometry_spring_authority_screen', 'physical': False,
              'status': 'inconclusive', 'solver_calls': 0, 'rows': []}
    base = solve_subsystem(tendon_map, contact_map, target, limits, friction, rows)
    result['solver_calls'] += 1
    result['base_status'] = base['status']
    if base['status'] == 'feasible':
        result['status'] = 'already_feasible'
        return result
    if base['status'] != INFEASIBLE:
        result['status'] = 'unresolved'
        result['base'] = _reduced(base)
        return result
    points = PROTOCOL['spring_shift_points']
    for row_index, dof in enumerate(passive):
        lo = ranges[row_index][0]+PROTOCOL['joint_limit_inset_rad']
        hi = ranges[row_index][1]-PROTOCOL['joint_limit_inset_rad']
        entry = {'dof': int(dof), 'qpos_adr': int(qpos_adrs[row_index]),
                 'stiffness': float(stiffness[row_index]),
                 'springref': float(springref[row_index]),
                 'range': [float(v) for v in ranges[row_index]],
                 'current_q': float(q_support[qpos_adrs[row_index]]),
                 'can_resolve': False, 'points': []}
        for q_new in np.linspace(lo, hi, points):
            if time.monotonic() >= deadline:
                result['status'] = 'budget_exhausted'
                result['rows'].append(entry)
                return result
            if result['solver_calls'] >= max_calls:
                result['status'] = 'call_budget_exhausted'
                result['rows'].append(entry)
                return result
            adjusted = np.asarray(target, dtype=float).copy()
            adjusted[dof] += stiffness[row_index]*(q_new-entry['current_q'])
            report = solve_support(np.asarray(tendon_map)[rows],
                                   np.asarray(contact_map)[rows],
                                   adjusted[rows], limits, friction)
            result['solver_calls'] += 1
            status = report['status']
            point = {'q': float(q_new), 'delta': float(stiffness[row_index]*(q_new-entry['current_q'])),
                     'status': status}
            if 'nearest_balance' in report:
                point['minimum_scaled_balance_error'] = report['nearest_balance'].get(
                    'minimum_scaled_balance_error')
            entry['points'].append(point)
            if status == 'feasible':
                entry['can_resolve'] = True
                break
        result['rows'].append(entry)
    result['status'] = 'completed'
    result['rows_that_can_resolve'] = int(sum(e['can_resolve'] for e in result['rows']))
    return result


def _evaluate_trial(body, transmission, q_trial, fmax, l0, *, deadline, witness):
    """Run the fixed gate chain for one declared posture; source is never mutated.

    witness is a shared mutable counter bounding how many support-matrix sets
    are saved per candidate; a feasible gate result always forces a save so a
    positive claim can never lack replayable evidence.
    """
    import mujoco as mj
    from .pose_probe import first_contact_height
    from .support_diagnostics import contact_columns, evaluate_support
    from .force_audit import audit_static_forces
    from .foot_placement import root_balance, select_sample
    model, data = body.m, body.d
    arrays = {}
    record = {'solver_calls': 0, 'samples': [], 'status': 'pending'}
    height = first_contact_height(model, data, q_trial, deadline=deadline)
    record['first_contact_height_native'] = float(height)
    for n, depth in enumerate(PROTOCOL['depths_native']):
        _deadline(deadline)
        data.qpos[:] = q_trial; data.qpos[2] = height-depth
        data.qvel[:] = 0; data.qacc_warmstart[:] = 0
        data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        if np.any(data.warning.number):
            raise ValueError('Solver warning in passive-posture probe')
        C, mu, contacts, nonfoot = contact_columns(model, data)
        legs = [leg for leg in LEGS if any('/'+leg+'_tarsus' in c['geom'] for c in contacts)]
        sample = {'index': n, 'depth_native': depth, 'foot_legs': legs,
                  'nonfoot_contacts': nonfoot, 'contacts': contacts,
                  'root_balance': {'feasible': False, 'status': 'not_eligible'}}
        sp = f'depth_{n:02d}_'
        arrays.update({sp+'qpos': data.qpos.copy(), sp+'contact_map': C[:6].copy(),
                       sp+'target': (data.qfrc_bias-data.qfrc_passive)[:6].copy(),
                       sp+'friction': mu.copy()})
        if contacts and not nonfoot:
            sample['root_balance'] = root_balance(arrays[sp+'contact_map'],
                                                  arrays[sp+'target'], mu)
            record['solver_calls'] += 1
        record['samples'].append(sample)
    selected = select_sample(record['samples'])
    record['selected_sample_index'] = None if selected is None else selected['index']
    for sample in record['samples']:
        if sample['root_balance'].get('feasible') is not True:
            continue
        _deadline(deadline)
        sp = f'depth_{sample["index"]:02d}_'
        data.qpos[:] = arrays[sp+'qpos']; data.qvel[:] = 0
        data.qacc_warmstart[:] = 0; data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        limits = fmax*np.exp(-((data.ten_length/l0-1)/.5)**2)
        support, inputs = evaluate_support(body, transmission, limits)
        record['solver_calls'] += 1
        rows = passive_rows(inputs['support_tendon_map'], range(6))
        subsystem = solve_subsystem(*(inputs[k] for k in FIELDS[:4]),
                                    inputs['support_friction'], list(range(6))+rows)
        record['solver_calls'] += 1
        sample['passive_rows'] = rows
        sample['passive_subsystem'] = _reduced(subsystem)
        sample['support'] = _reduced(support)
        feasible = subsystem['status'] == 'feasible' or support['status'] == FEASIBLE_POSE
        if feasible or witness[0] < PROTOCOL['max_saved_support_witnesses_per_candidate']:
            witness[0] += 1
            audit, forces = audit_static_forces(model, data.qpos)
            if not audit['passed'] or not np.allclose(forces['support_target'],
                        inputs['support_target'], rtol=1e-12, atol=1e-12):
                raise ValueError('Trial support force accounting mismatch')
            sample['force_accounting'] = audit
            arrays.update({sp+k: v for k, v in inputs.items()})
            arrays[sp+'tendon_lengths'] = data.ten_length.copy()
            arrays.update({sp+'force_'+k: v for k, v in forces.items()})
            sample['evidence_saved'] = True
        else:
            # A positive gate result always took the save branch above.
            sample['evidence_saved'] = False
    best = 0
    for sample in record['samples']:
        gate = 0
        if sample['foot_legs'] and not sample['nonfoot_contacts']:
            gate = 1
            if sample['root_balance'].get('feasible') is True:
                gate = 2
                if sample.get('passive_subsystem', {}).get('status') == 'feasible':
                    gate = 3
                    if sample.get('support', {}).get('status') == FEASIBLE_POSE:
                        gate = 4
        sample['gate_index'] = gate
        best = max(best, gate)
    record['best_gate'] = best
    record['best_gate_name'] = GATE_ORDER[best-1] if best else 'no_foot_only_contact'
    record['any_passive_feasible'] = any(
        s.get('passive_subsystem', {}).get('status') == 'feasible' for s in record['samples'])
    record['any_full_feasible'] = any(
        s.get('support', {}).get('status') == FEASIBLE_POSE for s in record['samples'])
    if selected is None:
        record['status'] = 'no_foot_only_pose'
    elif selected['root_balance'].get('feasible') is not True:
        record['status'] = 'root_balance_unresolved'
    elif 'passive_subsystem' not in selected:
        record['status'] = 'passive_gate_not_run'
    else:
        status = selected['passive_subsystem']['status']
        if status != 'feasible':
            record['status'] = 'passive_rows_'+status
        elif 'support' not in selected:
            record['status'] = 'full_gate_not_run'
        else:
            record['status'] = 'full_support_'+selected['support']['status']
    return record, arrays


def run(workspace):
    """Bounded posture search on the retained placements; new physics, same model."""
    import mujoco as mj
    from .body import Body
    from .foot_placement import verify as verify_placement
    from .root_ab import derive_xml, audit_models, file_hash, write_json
    from .root_ab_evidence import read_json
    from .transmission import SiteTendonTransmission
    workspace = Path(workspace).resolve()
    out = workspace/'passive-posture'; out.mkdir(exist_ok=False)
    placed_dir = workspace/'foot-placement'; source_dir = workspace/'source-study'
    started = time.monotonic(); deadline = started+PROTOCOL['worker_wall_s']
    result = {'schema': 1, 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
              'status': 'running', 'scientific_outcome': 'not_run', 'candidates': [],
              'biological_validation': False, 'walking_claimed': False,
              'standing_claimed': False, 'CNS_executed': False,
              'dynamic_trials_executed': 0, 'new_physics_trajectories': 0}
    arrays = {}
    try:
        record_path = Path(__file__).resolve().parents[1]/'examples/fe02-passive-posture.json'
        if read_json(record_path)['parameters'] != PROTOCOL:
            raise ValueError('Versioned record differs from implemented protocol')
        inputs = [placed_dir/'result.json', placed_dir/'observations.npz',
                  placed_dir/'model-receipt.json', workspace/'protocol.json',
                  source_dir/'result.json', source_dir/'observations.npz']
        record_inputs = {p.relative_to(workspace).as_posix(): file_hash(p) for p in inputs}
        code = {name: file_hash(Path(__file__).with_name(name)) for name in
                ('passive_posture.py', 'static_support.py', 'foot_placement.py',
                 'support_diagnostics.py', 'pose_probe.py', 'root_ab_evidence.py')}
        frozen = {'schema': 1, 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
                  'input_sha256': record_inputs, 'code_sha256': code,
                  'versions': {n: importlib.metadata.version(n)
                               for n in ('numpy', 'scipy', 'mujoco', 'flygym')}}
        # The immutable execution identity and all settings precede data collection.
        write_json(out/'protocol.json', frozen)
        verification = verify_placement(workspace)
        if verification.get('evidence_valid') is not True:
            raise ValueError('Placement source evidence was not verified')
        result['source_verification_sha256'] = file_hash(placed_dir/'verification.json')
        source = read_json(source_dir/'result.json')
        placed = read_json(placed_dir/'result.json')
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
                      placement_result_sha256=record_inputs['foot-placement/result.json'],
                      placement_observations_sha256=record_inputs['foot-placement/observations.npz'],
                      input_sha256=record_inputs, code_sha256=code,
                      versions=frozen['versions'], units=source['units'])
        body = Body.__new__(Body); body.m = model; body.d = mj.MjData(model)
        data = body.d; transmission = SiteTendonTransmission(model)
        with np.load(source_dir/'observations.npz', allow_pickle=False) as z:
            q0 = z['neutral_qpos'].copy()
        with np.load(placed_dir/'observations.npz', allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        if q0.shape != (model.nq,) or not np.isfinite(q0).all():
            raise ValueError('Finite matching neutral coordinates required')
        data.qpos[:] = q0; data.qvel[:] = 0
        data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        l0 = data.ten_length.copy()
        fmax = np.asarray(source['maximum_tensions_native'], dtype=float)
        if (model.ntendon != 90 or model.nu != 0 or fmax.shape != (90,)
                or not np.isfinite(fmax).all() or np.any(fmax <= 0) or np.any(l0 <= 0)):
            raise ValueError('Original unactuated 90-tendon model and positive bounds required')
        arrays.update(neutral_qpos=q0, neutral_lengths=l0, maximum_tensions=fmax)
        for candidate in placed['candidates']:
            if time.monotonic() >= deadline:
                raise TimeoutError('Passive-posture worker wall budget exceeded')
            index = candidate['index']; prefix = f'pose_{index:02d}_'
            entry = {'index': index, 'placement_status': candidate['status'],
                     'trials': [], 'solver_calls': 0}
            result['candidates'].append(entry)
            if 'support' not in candidate:
                entry['status'] = 'no_root_feasible_placement'
                continue
            q_candidate = saved[prefix+'candidate_qpos']
            root_dofs = candidate['force_accounting']['root_dofs']
            T_saved = saved[prefix+'support_tendon_map']
            passive = passive_rows(T_saved, root_dofs)
            data.qpos[:] = saved[prefix+'support_qpos']; data.qvel[:] = 0
            data.qacc_warmstart[:] = 0; data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
            mj.mj_forward(model, data)
            if passive_rows(transmission.matrix(data), root_dofs) != passive:
                raise ValueError('Passive row set differs between saved and rebuilt model')
            joint_ids = np.array([int(model.dof_jntid[d]) for d in passive])
            qpos_adrs = model.jnt_qposadr[joint_ids]
            ranges = model.jnt_range[joint_ids]
            stiffness = model.jnt_stiffness[joint_ids]
            springref = model.qpos_spring[qpos_adrs]
            arrays[prefix+'passive_dofs'] = np.asarray(passive, dtype=int)
            arrays[prefix+'passive_qpos_adrs'] = qpos_adrs
            arrays[prefix+'passive_ranges'] = ranges
            arrays[prefix+'passive_stiffness'] = stiffness
            arrays[prefix+'passive_springref'] = springref
            entry['passive_dofs'] = [int(d) for d in passive]
            entry['passive_joint_names'] = [
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(j)) for j in joint_ids]
            remaining = deadline-time.monotonic()
            entry['spring_screen'] = _spring_screen(
                *(saved[prefix+k] for k in FIELDS), root_rows=root_dofs, passive=passive,
                support_qpos=saved[prefix+'support_qpos'], qpos_adrs=qpos_adrs,
                stiffness=stiffness, springref=springref, ranges=ranges,
                wall_seconds=min(30., max(1., remaining)),
                max_calls=PROTOCOL['screen_solver_calls_per_candidate'])
            trials = trial_postures(q_candidate, qpos_adrs, ranges, index)
            entry['declared_trials'] = len(trials)+PROTOCOL['composed_trials']
            candidate_deadline = min(deadline, time.monotonic()+PROTOCOL['candidate_wall_s'])
            witness = [0]

            def unexecuted_marker(status):
                entry['trials'].append({'meta': {'kind': 'not_run'}, 'status': status,
                                        'remaining_trials_unexecuted': True})

            for q_trial, meta in trials:
                if time.monotonic() >= candidate_deadline:
                    unexecuted_marker('budget_exhausted'); break
                if entry['solver_calls'] >= PROTOCOL['max_solver_calls_per_candidate']:
                    unexecuted_marker('call_budget_exhausted'); break
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
                tp = prefix+f'trial_{len(entry["trials"]):03d}_'
                arrays.update({tp+k: v for k, v in trial_arrays.items()})
                arrays[tp+'qpos'] = np.asarray(q_trial, dtype=float)
                record['meta'] = meta
                entry['trials'].append(record)
                entry['solver_calls'] += record['solver_calls']
            for _ in range(PROTOCOL['composed_trials']):
                if len(entry['trials']) == entry['declared_trials']:
                    break
                if entry['trials'] and entry['trials'][-1]['meta']['kind'] == 'not_run':
                    break
                if time.monotonic() >= candidate_deadline:
                    unexecuted_marker('budget_exhausted'); break
                if entry['solver_calls'] >= PROTOCOL['max_solver_calls_per_candidate']:
                    unexecuted_marker('call_budget_exhausted'); break
                q_composed, meta = compose_trial(q_candidate, entry['trials'])
                try:
                    record, trial_arrays = _evaluate_trial(
                        body, transmission, q_composed, fmax, l0,
                        deadline=candidate_deadline, witness=witness)
                except ValueError as exc:
                    if 'not bracketed' in str(exc):
                        entry['trials'].append({'meta': meta, 'status': 'not_bracketed',
                                                'reason': str(exc)})
                        continue
                    raise
                tp = prefix+f'trial_{len(entry["trials"]):03d}_'
                arrays.update({tp+k: v for k, v in trial_arrays.items()})
                arrays[tp+'qpos'] = np.asarray(q_composed, dtype=float)
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
        if after != record_inputs:
            raise ValueError('Physical source evidence changed during the search')
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite passive-posture observations')
        _deadline(deadline)
        n_full = sum(c.get('full_feasible_trials', 0) for c in result['candidates'])
        n_passive = sum(c.get('passive_feasible_trials', 0) for c in result['candidates'])
        outcome = ('full_support_feasible_posture_found' if n_full else
                   'passive_feasible_without_full_support' if n_passive else
                   'no_passive_equilibrium_in_bounded_search')
        result.update(status='completed', scientific_outcome=outcome,
                      candidates_processed=len(result['candidates']))
    except Exception as exc:
        result.update(status='failed', scientific_outcome=(
            'inconclusive' if isinstance(exc, (TimeoutError, MemoryError))
            else 'blocked' if isinstance(exc, ModuleNotFoundError) else 'invalid_experiment'),
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
    """Replay the saved search evidence without MuJoCo; fail closed on mismatch."""
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import read_json, require, check_force
    from .foot_placement import (root_balance, root_contact_map, select_sample,
                                 validate_root_result, validate_sample)
    workspace = Path(workspace); out = workspace/'passive-posture'
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
                and result['new_physics_trajectories'] == 0, 'unsupported capability claim')
        require(result.get('protocol') == PROTOCOL
                and result.get('protocol_sha256') == protocol_hash(), 'protocol mismatch')
        frozen = read_json(out/'protocol.json')
        require(frozen.get('protocol') == PROTOCOL
                and frozen.get('protocol_sha256') == protocol_hash()
                and frozen.get('input_sha256') == result.get('input_sha256'),
                'pre-execution protocol record mismatch')
        placed = read_json(workspace/'foot-placement/result.json')
        for rel, recorded in (result.get('input_sha256') or {}).items():
            require(file_hash(workspace/rel) == recorded, 'input evidence changed: '+rel)
        require(result['placement_result_sha256']
                == file_hash(workspace/'foot-placement/result.json')
                and result['placement_observations_sha256']
                == file_hash(workspace/'foot-placement/observations.npz'),
                'placement evidence changed')
        if (result['status'] != 'completed'
                or result.get('observations', {}).get('sha256')
                != file_hash(out/'observations.npz')):
            raise ValueError('Incomplete result or observation hash mismatch')
        with np.load(out/'observations.npz', allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite saved observations')
        source = read_json(workspace/'source-study/result.json')
        require(result['versions'] == source['versions'] and result['units'] == source['units'],
                'source environment or units changed')
        receipt = read_json(workspace/'foot-placement/model-receipt.json')
        identity = receipt.pop('model_identity')
        require(identity == result['model_identity']
                and identity == hashlib.sha256(
                    json.dumps(receipt, sort_keys=True).encode()).hexdigest()
                and receipt['parent_body_sha256'] == source['body_sha256'],
                'derived model identity mismatch')
        np.testing.assert_array_equal(arrays['maximum_tensions'],
                                      source['maximum_tensions_native'])
        with np.load(workspace/'foot-placement/observations.npz', allow_pickle=False) as pz:
            placement_arrays = {k: pz[k].copy() for k in pz.files}
        np.testing.assert_array_equal(arrays['neutral_qpos'], placement_arrays['neutral_qpos'])
        trials_checked = witnesses_checked = 0
        passive_feasible = full_feasible = 0
        inconclusive = False
        for entry in result['candidates']:
            index = entry['index']; prefix = f'pose_{index:02d}_'
            candidate = placed['candidates'][index]
            require(candidate['index'] == index, 'candidate ordering changed')
            if entry.get('status') == 'no_root_feasible_placement':
                require('support' not in candidate and not entry['trials'],
                        'skipped candidate actually had support')
                continue
            require(entry['status'] == 'completed', 'candidate record incomplete')
            passive = [int(i) for i in arrays[prefix+'passive_dofs']]
            qpos_adrs = arrays[prefix+'passive_qpos_adrs']
            ranges = arrays[prefix+'passive_ranges']
            stiffness = arrays[prefix+'passive_stiffness']
            springref = arrays[prefix+'passive_springref']
            require(entry['passive_dofs'] == passive, 'recorded passive DOF list mismatch')
            T_saved = placement_arrays[prefix+'support_tendon_map']
            require(passive == [i for i in np.flatnonzero(np.all(T_saved == 0, axis=1))
                                if i not in candidate['force_accounting']['root_dofs']],
                    'saved passive rows are not the exact-zero rows')
            q_candidate = placement_arrays[prefix+'candidate_qpos']
            require(qpos_adrs.ndim == 1 and len(qpos_adrs) == len(passive)
                    and len(set(qpos_adrs.tolist())) == len(qpos_adrs)
                    and np.all(qpos_adrs >= 7) and np.all(qpos_adrs < len(q_candidate))
                    and ranges.shape == (len(passive), 2)
                    and np.all(ranges[:, 0] < ranges[:, 1])
                    and np.all(q_candidate[qpos_adrs] >= ranges[:, 0])
                    and np.all(q_candidate[qpos_adrs] <= ranges[:, 1]),
                    'invalid saved passive coordinates or limits')
            # Tier A screen: replay every recorded point on the saved matrices.
            screen = entry['spring_screen']
            require(screen.get('kind') == 'fixed_geometry_spring_authority_screen'
                    and screen.get('physical') is False, 'screen mislabeled as physical')
            root_dofs = candidate['force_accounting']['root_dofs']
            rows = list(root_dofs)+passive
            base = solve_subsystem(*(placement_arrays[prefix+k] for k in FIELDS[:4]),
                                   placement_arrays[prefix+'support_friction'], rows)
            require(screen.get('base_status') == base['status'], 'screen base mismatch')
            if screen['status'] == 'already_feasible':
                require(base['status'] == 'feasible', 'false screen feasibility')
            elif screen['status'] in ('completed', 'budget_exhausted', 'call_budget_exhausted'):
                require(base['status'] == INFEASIBLE, 'screen ran on a non-infeasible base')
                for row_entry in screen['rows']:
                    seen = []
                    dof = row_entry['dof']; adr = row_entry['qpos_adr']
                    r_i = passive.index(dof)
                    require(qpos_adrs[r_i] == adr
                            and np.isclose(row_entry['stiffness'], stiffness[r_i], rtol=0, atol=0)
                            and np.isclose(row_entry['springref'], springref[r_i], rtol=0, atol=0)
                            and row_entry['range'] == [float(v) for v in ranges[r_i]]
                            and np.isclose(row_entry['current_q'],
                                           placement_arrays[prefix+'support_qpos'][adr],
                                           rtol=0, atol=0), 'screen row metadata mismatch')
                    lo = ranges[r_i][0]+PROTOCOL['joint_limit_inset_rad']
                    hi = ranges[r_i][1]-PROTOCOL['joint_limit_inset_rad']
                    expected_q = np.linspace(lo, hi, PROTOCOL['spring_shift_points'])
                    require(len(row_entry['points']) <= len(expected_q), 'extra screen points')
                    for p_i, point in enumerate(row_entry['points']):
                        require(np.isclose(point['q'], expected_q[p_i], rtol=0, atol=0)
                                and np.isclose(point['delta'],
                                               stiffness[r_i]*(point['q']-row_entry['current_q']),
                                               rtol=0, atol=0), 'screen point mismatch')
                        adjusted = placement_arrays[prefix+'support_target'].copy()
                        adjusted[dof] += point['delta']
                        replay = solve_support(T_saved[rows],
                                               placement_arrays[prefix+'support_contact_map'][rows],
                                               adjusted[rows],
                                               placement_arrays[prefix+'support_limits'],
                                               placement_arrays[prefix+'support_friction'])
                        require(point['status'] == replay['status'], 'screen point not reproducible')
                        inconclusive |= replay['status'] == 'inconclusive'
                        seen.append(replay['status'] == 'feasible')
                    # Feasible points stop the sweep, so feasibility may only be last.
                    require(all(flag is False for flag in seen[:-1])
                            and row_entry['can_resolve'] is (bool(seen) and seen[-1]),
                            'screen resolve flag mismatch')
                if screen['status'] == 'completed':
                    require(screen.get('rows_that_can_resolve')
                            == sum(e['can_resolve'] for e in screen['rows']),
                            'screen summary count mismatch')
            else:
                require(screen['status'] == 'unresolved', 'unknown screen status')
                inconclusive = True
            # Tier B: regenerated declared trial order must match recorded metadata.
            expected = trial_postures(q_candidate, qpos_adrs, ranges, index)
            trials = entry['trials']
            require(len(trials) <= entry['declared_trials']
                    == len(expected)+PROTOCOL['composed_trials'],
                    'declared trial count mismatch')
            declared = [m for _, m in expected]+[{'kind': 'composed'}]*PROTOCOL['composed_trials']
            for t_i, trial in enumerate(trials):
                meta = trial['meta']
                if meta['kind'] == 'not_run':
                    require(t_i == len(trials)-1
                            and trial.get('remaining_trials_unexecuted') is True
                            and trial['status'] in ('budget_exhausted', 'call_budget_exhausted'),
                            'unexecuted trials must end the record explicitly')
                    inconclusive = True
                    continue
                require(t_i < len(declared) and meta['kind'] == declared[t_i]['kind'],
                        'trial order or kind differs from the declared list')
                tp = prefix+f'trial_{t_i:03d}_'
                if meta['kind'] == 'composed':
                    declared_q, regenerated_meta = compose_trial(q_candidate, trials[:t_i])
                    require(meta == regenerated_meta, 'composed trial metadata changed')
                else:
                    matches = [q for q, m in expected if m == meta]
                    require(len(matches) == 1, 'declared trial metadata is not unique')
                    declared_q = matches[0]
                fixed = np.ones(len(declared_q), dtype=bool); fixed[qpos_adrs] = False
                np.testing.assert_array_equal(declared_q[fixed], q_candidate[fixed])
                require(np.all(declared_q[qpos_adrs] >= ranges[:, 0])
                        and np.all(declared_q[qpos_adrs] <= ranges[:, 1]),
                        'trial outside original joint limits')
                if trial['status'] == 'not_bracketed':
                    require(not trial.get('samples')
                            and not any(k.startswith(tp) for k in arrays),
                            'unbracketed trial has samples or arrays')
                    continue
                np.testing.assert_array_equal(arrays[tp+'qpos'], declared_q)
                q_trial = declared_q
                samples = trial['samples']
                require([s['index'] for s in samples] == list(range(len(PROTOCOL['depths_native'])))
                        and [s['depth_native'] for s in samples] == PROTOCOL['depths_native'],
                        'incomplete fixed depth grid')
                trials_checked += 1
                for sample in samples:
                    sp = tp+f'depth_{sample["index"]:02d}_'
                    q = q_trial.copy()
                    q[2] = trial['first_contact_height_native']-sample['depth_native']
                    np.testing.assert_array_equal(arrays[sp+'qpos'], q)
                    validate_sample(sample, arrays[sp+'contact_map'], arrays[sp+'friction'])
                    np.testing.assert_allclose(arrays[sp+'contact_map'],
                                               root_contact_map(q, sample['contacts']),
                                               rtol=1e-12, atol=1e-12)
                    if sample['foot_legs'] and not sample['nonfoot_contacts']:
                        validate_root_result(sample['root_balance'], arrays[sp+'contact_map'],
                                             arrays[sp+'target'], arrays[sp+'friction'])
                        recalculated = root_balance(arrays[sp+'contact_map'],
                                                    arrays[sp+'target'], arrays[sp+'friction'])
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
                                                       arrays[sp+'target'], rtol=1e-12, atol=1e-12)
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
                                                        if recomputed_best else 'no_foot_only_contact'),
                        'trial best gate mismatch')
                require(trial['any_passive_feasible'] == any(
                    s.get('passive_subsystem', {}).get('status') == 'feasible' for s in samples)
                    and trial['any_full_feasible'] == any(
                        s.get('support', {}).get('status') == FEASIBLE_POSE for s in samples),
                    'trial feasibility flags mismatch')
                require(trial['solver_calls'] >= 0
                        and type(trial['solver_calls']) is int, 'invalid solver call count')
            require(type(entry.get('solver_calls')) is int and entry['solver_calls'] >= 0
                    and entry['solver_calls'] == sum(t.get('solver_calls', 0) for t in trials)
                    and entry.get('trials_executed') == sum('best_gate' in t for t in trials)
                    and entry.get('passive_feasible_trials')
                    == sum(t.get('any_passive_feasible') is True for t in trials)
                    and entry.get('full_feasible_trials')
                    == sum(t.get('any_full_feasible') is True for t in trials)
                    and entry.get('best_gate')
                    == max([t['best_gate'] for t in trials if 'best_gate' in t], default=0),
                    'candidate summary mismatch')
        outcome = result['scientific_outcome']
        expected_outcome = ('full_support_feasible_posture_found' if full_feasible else
                            'passive_feasible_without_full_support' if passive_feasible else
                            'no_passive_equilibrium_in_bounded_search')
        require(outcome == expected_outcome, 'recorded outcome does not match replayed gates')
        verdict.update(status='completed', evidence_valid=True,
                       scientific_outcome='inconclusive' if inconclusive else outcome,
                       trials_checked=trials_checked, witnesses_checked=witnesses_checked,
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
    report = verify(args.workspace) if args.verify else run(args.workspace)
    print(json.dumps({k: v for k, v in report.items() if k != 'candidates'}, indent=2))
