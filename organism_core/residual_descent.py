"""Residual-descent posture generation over every non-root DOF (FE-02).

The bounded experiments decomposed the static-support deficit to its final
lever: root+contact balance is attainable, the passive subsystem contracts
asymptotically under margin-directed passive shifts, and unbounded tendon
tension cannot repair the remaining rows. The only unexplored posture space
is the actuated (tendon-spanned) joint set, which no prior protocol moved.

Two declared phases, one generator and one certifier:

Phase 1 (fixed-geometry descent) minimizes R(q) = the passive-subsystem
nearest-balance scaled error over all non-root DOFs, holding the saved
contact map, tendon map and friction of a contracting iterative-shift seed
fixed while only the bias target b(q) = qfrc_bias - qfrc_passive is
recomputed. Reaching R ~ 0 here is an algebraic statement about the saved
contact geometry, not a physical posture claim.

Phase 2 (physical certification) re-derives contacts and runs the unchanged
gate chain (foot-only contact -> root balance -> passive subsystem -> full
support, with force audit and margin LP) on the descent terminal poses.
Only certified gate outcomes carry any physical meaning.

Static witness generation only — no dynamic hold, standing, walking,
flight, CNS execution or biological validation.
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
from .passive_posture import FIELDS, solve_subsystem, INFEASIBLE, FEASIBLE_POSE
from .passive_margin import margin_solve, achievable_intervals, passive_rows

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PROTOCOL = {
    'id': 'FE-02-residual-descent-v1',
    'seed_margin_threshold': 1e-3,
    'max_seeds_total': 8,
    'descent_dofs': ('every non-root DOF row (actuated and passive); the six '
                     'free-root DOF rows are held fixed'),
    'objective': ('passive-subsystem minimum scaled balance error on the '
                  'saved contact map, tendon map and friction; only the '
                  'bias target is recomputed at the trial pose'),
    'descent_rule': ('deterministic cyclic coordinate descent in row order; '
                     'per-DOF step h_rel*(joint range), h_rel starts at 0.25 '
                     'and halves after any sweep without improvement'),
    'h_rel_initial': 0.25, 'h_rel_floor': 1e-4,
    'joint_limit_inset_rad': 1e-4,
    'max_evals_per_seed': 3000,
    'certified_poses_per_seed': 2,
    'depths_native': [.001, .005, .01, .02, .04],
    'bracket_half_range_native': 2., 'bracket_iterations': 25,
    'root_solver_time_limit_s': 2.,
    'max_solver_calls_per_seed': 4000,
    'seed_wall_s': 240., 'worker_wall_s': 900., 'memory_mib': 8192,
    'replay_tolerance': 1e-6,
    'outcomes': ['found_full_support_pose', 'found_subsystem_feasible_pose',
                 'descent_floored_without_feasibility', 'inconclusive',
                 'invalid_experiment'],
    'verification': ('replay of saved-input LPs in the same SciPy/HiGHS '
                     'family; no independent physics or biological check'),
    'scope': ('Static witness generation under declared approximations; '
              'certification re-derives contacts but claims no dynamic '
              'hold, standing, walking or biological result'),
}
WITNESS = re.compile(r'^(desc_\d+_cert_\d+_depth_\d+_)support_tendon_map$')


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _residual(T, C, target, limits, friction, rows):
    """Subsystem min scaled balance error; 0 when the LP is feasible.

    Returns (R, status). An undecided solver state yields (inf, status) and
    is counted by the caller; it is never treated as feasibility.
    """
    report = solve_subsystem(T, C, target, limits, friction, rows)
    if report['status'] == 'feasible':
        return 0., 'feasible'
    near = report.get('nearest_balance')
    if report['status'] == INFEASIBLE and near and np.isfinite(
            near.get('minimum_scaled_balance_error', np.inf)):
        return float(near['minimum_scaled_balance_error']), 'infeasible'
    return float('inf'), report.get('status', 'unresolved')


def _descend(model, data, T, C, friction, limits, rows, dof_rows, dof_adrs,
             dof_ranges, q_seed, budget):
    """Cyclic coordinate descent on the fixed-geometry subsystem residual.

    Every accepted step records (qpos, target, R) so the verifier can replay
    the trajectory without any physics. Rejected probes are counted, not
    saved. Returns (record, accepted arrays).
    """
    import mujoco as mj
    q = np.asarray(q_seed, dtype=float).copy()

    def target_at(qq):
        data.qpos[:] = qq; data.qvel[:] = 0
        data.qacc_warmstart[:] = 0; data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        if np.any(data.warning.number):
            raise ValueError('Solver warning in descent probe')
        return data.qfrc_bias - data.qfrc_passive

    b = target_at(q)
    R, status = _residual(T, C, b, limits, friction, rows)
    budget[0] -= 1
    _r = lambda v: float(v) if np.isfinite(v) else None
    rec = {'solver_calls': 1, 'forward_calls': 1, 'unresolved_evals': 0,
           'sweeps': [], 'status': 'pending', 'initial_R': _r(R),
           'initial_status': status}
    accepted = {}
    n_acc = 0
    accepted[f'acc_{n_acc:04d}_qpos'] = q.copy()
    accepted[f'acc_{n_acc:04d}_target'] = np.asarray(b, dtype=float)
    acc_rec = [{'accepted': 0, 'dof_row': -1, 'direction': 0, 'R': _r(R)}]
    h_rel = PROTOCOL['h_rel_initial']
    while budget[0] > 0 and h_rel >= PROTOCOL['h_rel_floor']:
        sweep = {'h_rel': h_rel, 'improvements': 0}
        for di, (adr, (lo, hi)) in enumerate(zip(dof_adrs, dof_ranges)):
            if budget[0] <= 0:
                break
            h = h_rel*(hi-lo)
            inset = PROTOCOL['joint_limit_inset_rad']
            best_R, best_q, best_b, best_sign = R, None, None, 0
            for sign in (1., -1.):
                if budget[0] <= 0:
                    break
                qp = q.copy()
                qp[adr] = min(max(q[adr]+sign*h, lo+inset), hi-inset)
                if qp[adr] == q[adr]:
                    continue
                bp = target_at(qp)
                Rp, st = _residual(T, C, bp, limits, friction, rows)
                budget[0] -= 1
                rec['solver_calls'] += 1; rec['forward_calls'] += 1
                if st not in ('feasible', 'infeasible'):
                    rec['unresolved_evals'] += 1
                if Rp < best_R:
                    best_R, best_q, best_b, best_sign = Rp, qp, bp, sign
            if best_q is not None:
                q, b, R = best_q, best_b, best_R
                n_acc += 1
                accepted[f'acc_{n_acc:04d}_qpos'] = q.copy()
                accepted[f'acc_{n_acc:04d}_target'] = np.asarray(b, dtype=float)
                acc_rec.append({'accepted': n_acc, 'dof_row': int(dof_rows[di]),
                                'direction': int(best_sign), 'R': _r(R)})
                sweep['improvements'] += 1
                if R == 0.:
                    break
        sweep['R_after'] = _r(R)
        rec['sweeps'].append(sweep)
        if R == 0.:
            rec['status'] = 'residual_zero_under_fixed_geometry'
            break
        if sweep['improvements'] == 0:
            h_rel *= .5
    else:
        if rec['status'] == 'pending':
            rec['status'] = ('eval_budget_exhausted' if budget[0] <= 0
                             else 'step_floor_reached')
    if rec['status'] == 'pending':
        rec['status'] = ('eval_budget_exhausted' if budget[0] <= 0
                         else 'step_floor_reached')
    rec.update(final_R=_r(R), accepted_steps=n_acc, accepted=acc_rec,
               terminal_qpos=q.tolist())
    return rec, accepted, q


def _certify(body, transmission, q_pose, fmax, l0, passive, qpos_adrs,
             stiffness, ranges, deadline, prefix):
    """Full physical gate chain at one generated pose; mirrors _iterate's
    per-iterate evaluation: fresh contact heights, depth sweep, contact
    re-derivation, root LP, subsystem LP, margin LP, full support, force
    audit. Returns (record, arrays)."""
    import mujoco as mj
    from .pose_probe import first_contact_height
    from .support_diagnostics import contact_columns, evaluate_support
    from .force_audit import audit_static_forces
    from .foot_placement import root_balance
    model, data = body.m, body.d
    rec = {'solver_calls': 0, 'samples': []}
    arrays = {}
    height = first_contact_height(model, data, q_pose, deadline=deadline)
    rec['first_contact_height_native'] = float(height)
    for n, depth in enumerate(PROTOCOL['depths_native']):
        data.qpos[:] = q_pose; data.qpos[2] = height-depth
        data.qvel[:] = 0; data.qacc_warmstart[:] = 0
        data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        if np.any(data.warning.number):
            raise ValueError('Solver warning in certification probe')
        C, mu, contacts, nonfoot = contact_columns(model, data)
        legs = [leg for leg in LEGS
                if any('/'+leg+'_tarsus' in c['geom'] for c in contacts)]
        sample = {'index': n, 'depth_native': depth, 'foot_legs': legs,
                  'nonfoot_contacts': nonfoot,
                  'root_balance': {'feasible': False, 'status': 'not_eligible'}}
        sp = f'{prefix}depth_{n:02d}_'
        arrays.update({sp+'qpos': data.qpos.copy(),
                       sp+'contact_map': C[:6].copy(),
                       sp+'target': (data.qfrc_bias-data.qfrc_passive)[:6].copy(),
                       sp+'friction': mu.copy()})
        if contacts and not nonfoot:
            sample['root_balance'] = root_balance(arrays[sp+'contact_map'],
                                                  arrays[sp+'target'], mu)
            rec['solver_calls'] += 1
        rec['samples'].append(sample)
    for sample in rec['samples']:
        if sample['root_balance'].get('feasible') is not True:
            sample['gate_index'] = (
                1 if sample['foot_legs'] and not sample['nonfoot_contacts']
                else 0)
            sample['evidence_saved'] = False
            continue
        n = sample['index']; sp = f'{prefix}depth_{n:02d}_'
        data.qpos[:] = arrays[sp+'qpos']; data.qvel[:] = 0
        data.qacc_warmstart[:] = 0; data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        limits = fmax*np.exp(-((data.ten_length/l0-1)/.5)**2)
        support, inputs = evaluate_support(body, transmission, limits)
        rec['solver_calls'] += 1
        subsystem = solve_subsystem(*(inputs[k] for k in FIELDS[:4]),
                                    inputs['support_friction'],
                                    list(range(6))+passive)
        rec['solver_calls'] += 1
        intervals = achievable_intervals(stiffness, ranges,
                                         data.qpos[qpos_adrs])
        margin = margin_solve(*(inputs[k] for k in FIELDS),
                              list(range(6)), passive, intervals)
        rec['solver_calls'] += 1
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
            sample['margin'].update(
                t=margin['t'],
                feasible_within_authority=margin['feasible_within_authority'],
                required_shift_native=margin['required_shift_native'])
        audit, forces = audit_static_forces(model, data.qpos)
        if not audit['passed'] or not np.allclose(
                forces['support_target'], inputs['support_target'],
                rtol=1e-12, atol=1e-12):
            raise ValueError('Certification support force accounting mismatch')
        sample['force_accounting'] = audit
        arrays.update({sp+k: v for k, v in inputs.items()})
        arrays[sp+'tendon_lengths'] = data.ten_length.copy()
        arrays.update({sp+'force_'+k: v for k, v in forces.items()})
        arrays[sp+'margin_intervals'] = intervals
        sample['evidence_saved'] = True
        if sample['support']['status'] == FEASIBLE_POSE:
            sample['gate_index'] = 4
        elif subsystem['status'] == 'feasible':
            sample['gate_index'] = 3
        else:
            sample['gate_index'] = 2
    rec['best_gate'] = max(s['gate_index'] for s in rec['samples'])
    rec['any_passive_feasible'] = any(
        s.get('passive_subsystem', {}).get('status') == 'feasible'
        for s in rec['samples'])
    rec['any_full_feasible'] = any(
        s.get('support', {}).get('status') == FEASIBLE_POSE
        for s in rec['samples'])
    return rec, arrays


def run(workspace):
    """Descend the fixed-geometry subsystem residual, then certify physically."""
    import mujoco as mj
    from .body import Body
    from .root_ab import derive_xml, audit_models, file_hash, write_json
    from .root_ab_evidence import read_json
    from .transmission import SiteTendonTransmission
    workspace = Path(workspace).resolve()
    out = workspace/'residual-descent'; out.mkdir(exist_ok=False)
    placed_dir = workspace/'foot-placement'
    posture_dir = workspace/'passive-posture'
    margin_dir = workspace/'passive-margin'
    shift_dir = workspace/'iterative-shift'
    source_dir = workspace/'source-study'
    started = time.monotonic(); deadline = started+PROTOCOL['worker_wall_s']
    result = {'schema': 1, 'protocol': PROTOCOL,
              'protocol_sha256': protocol_hash(), 'status': 'running',
              'scientific_outcome': 'not_run', 'seeds': [],
              'biological_validation': False, 'walking_claimed': False,
              'standing_claimed': False, 'CNS_executed': False,
              'dynamic_trials_executed': 0, 'new_physics_trajectories': 0}
    arrays = {}
    try:
        record_path = (Path(__file__).resolve().parents[1]
                       /'examples/fe02-residual-descent.json')
        if read_json(record_path)['parameters'] != PROTOCOL:
            raise ValueError('Versioned record differs from implemented protocol')
        shift_verdict = read_json(shift_dir/'verification.json')
        if shift_verdict.get('evidence_valid') is not True:
            raise ValueError('Iterative-shift evidence was not verified')
        inputs = [placed_dir/'result.json', placed_dir/'observations.npz',
                  placed_dir/'model-receipt.json', workspace/'protocol.json',
                  source_dir/'result.json', source_dir/'observations.npz',
                  posture_dir/'result.json', posture_dir/'observations.npz',
                  shift_dir/'result.json', shift_dir/'observations.npz',
                  shift_dir/'verification.json']
        record_inputs = {p.relative_to(workspace).as_posix(): file_hash(p)
                         for p in inputs}
        code = {name: file_hash(Path(__file__).with_name(name)) for name in
                ('residual_descent.py', 'iterative_shift.py',
                 'passive_margin.py', 'passive_posture.py',
                 'static_support.py', 'foot_placement.py',
                 'support_diagnostics.py', 'pose_probe.py',
                 'root_ab_evidence.py')}
        frozen = {'schema': 1, 'protocol': PROTOCOL,
                  'protocol_sha256': protocol_hash(),
                  'input_sha256': record_inputs, 'code_sha256': code,
                  'versions': {n: importlib.metadata.version(n)
                               for n in ('numpy', 'scipy', 'mujoco', 'flygym')}}
        write_json(out/'protocol.json', frozen)
        source = read_json(source_dir/'result.json')
        placed = read_json(placed_dir/'result.json')
        posture = read_json(posture_dir/'result.json')
        shift = read_json(shift_dir/'result.json')
        legacy = Body('muscle_compliance', 'tendon_candidate')
        if legacy.digest != source['body_sha256']:
            raise ValueError('Source model identity mismatch')
        xml = derive_xml(legacy.xml); model = mj.MjModel.from_xml_string(xml)
        invariants = audit_models(legacy.m, model)
        receipt = read_json(placed_dir/'model-receipt.json')
        expected = {'parent_body_sha256': legacy.digest,
                    'candidate_xml_sha256': hashlib.sha256(
                        xml.encode()).hexdigest(),
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
        with np.load(shift_dir/'observations.npz', allow_pickle=False) as z:
            s_arrays = {k: z[k].copy() for k in z.files}
        data.qpos[:] = q0; data.qvel[:] = 0
        data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        l0 = data.ten_length.copy()
        fmax = np.asarray(source['maximum_tensions_native'], dtype=float)
        if (model.ntendon != 90 or model.nu != 0 or fmax.shape != (90,)
                or not np.isfinite(fmax).all() or np.any(l0 <= 0)):
            raise ValueError(
                'Original unactuated 90-tendon model and positive bounds required')
        arrays.update(neutral_qpos=q0, neutral_lengths=l0,
                      maximum_tensions=fmax)
        # Non-root DOF -> qpos address and declared joint range.
        dof_rows, dof_adrs, dof_ranges = [], [], []
        for d in range(6, model.nv):
            j = int(model.dof_jntid[d])
            dof_rows.append(d)
            dof_adrs.append(int(model.jnt_qposadr[j]))
            dof_ranges.append((float(model.jnt_range[j, 0]),
                               float(model.jnt_range[j, 1])))
        arrays['descent_dof_rows'] = np.asarray(dof_rows, dtype=int)
        arrays['descent_dof_qpos_adrs'] = np.asarray(dof_adrs, dtype=int)
        arrays['descent_dof_ranges'] = np.asarray(dof_ranges, dtype=float)
        # Seeds: final-iterate driver witnesses whose margin t meets the
        # declared threshold, in (candidate, lineage) order.
        seeds = []
        for entry in shift['candidates']:
            for li, lin in enumerate(entry.get('lineages', [])):
                its = lin.get('iterates', [])
                last = its[-1] if its else {}
                t = last.get('driver_margin_t')
                drv = last.get('driver_sample')
                if (t is None or drv is None
                        or t > PROTOCOL['seed_margin_threshold']):
                    continue
                n = last['samples'][drv]['index']
                prefix = (f"pose_{entry['index']:02d}_lineage_{li:02d}"
                          f"_it_{len(its)-1:02d}_depth_{n:02d}_")
                seeds.append({'index': entry['index'], 'lineage': li,
                              'iterate': len(its)-1, 'driver_sample': drv,
                              'margin_t': t, 'prefix': prefix})
        seeds = seeds[:PROTOCOL['max_seeds_total']]
        result['declared_seeds'] = len(seeds)
        for si, seed in enumerate(seeds):
            if time.monotonic() >= deadline:
                result['seeds'].append({'meta': seed, 'status': 'not_run',
                                        'remaining_seeds_unexecuted': True})
                break
            seed_deadline = min(deadline,
                                time.monotonic()+PROTOCOL['seed_wall_s'])
            sp = seed['prefix']
            T = s_arrays[sp+'support_tendon_map']
            C = s_arrays[sp+'support_contact_map']
            friction = s_arrays[sp+'support_friction']
            limits = s_arrays[sp+'support_limits']
            q_seed = s_arrays[sp+'support_qpos'].copy()
            index = seed['index']; pp = f'pose_{index:02d}_'
            passive = [int(i) for i in t_arrays[pp+'passive_dofs']]
            padrs = t_arrays[pp+'passive_qpos_adrs']
            pranges = t_arrays[pp+'passive_ranges']
            pstiff = t_arrays[pp+'passive_stiffness']
            placed_cand = placed['candidates'][index]
            root_dofs = placed_cand['force_accounting']['root_dofs']
            if passive != passive_rows(p_arrays[pp+'support_tendon_map'],
                                       root_dofs):
                raise ValueError('Passive row set differs between saved matrices')
            rows = list(range(6))+passive
            srec = {'meta': seed, 'status': 'running', 'solver_calls': 0,
                    'certifications': []}
            budget = [PROTOCOL['max_evals_per_seed']]
            drec, acc_arrays, q_term = _descend(
                model, data, T, C, friction, limits, rows,
                dof_rows, dof_adrs, dof_ranges, q_seed, budget)
            srec['solver_calls'] += drec['solver_calls']
            srec['forward_calls'] = drec['forward_calls']
            srec['unresolved_evals'] = drec['unresolved_evals']
            srec['descent'] = {k: v for k, v in drec.items()
                               if k not in ('solver_calls', 'forward_calls')}
            dp = f'desc_{si:02d}_'
            arrays.update({dp+k: v for k, v in acc_arrays.items()})
            arrays[dp+'seed_qpos'] = np.asarray(q_seed, dtype=float)
            arrays[dp+'seed_contact_map'] = np.asarray(C, dtype=float)
            arrays[dp+'seed_tendon_map'] = np.asarray(T, dtype=float)
            arrays[dp+'seed_friction'] = np.asarray(friction, dtype=float)
            arrays[dp+'seed_limits'] = np.asarray(limits, dtype=float)
            arrays[dp+'seed_target'] = np.asarray(
                s_arrays[sp+'support_target'], dtype=float)
            # Certification poses: terminal pose and the accepted pose at
            # the midpoint of the residual drop (declared diversity pick).
            n_acc = drec['accepted_steps']
            cert_keys = {n_acc}
            if (n_acc > 1 and drec['initial_R'] is not None
                    and drec['final_R'] is not None):
                half = (drec['initial_R']+drec['final_R'])/2.
                mid = min((a for a in drec['accepted'] if a['R'] is not None),
                          key=lambda a: abs(a['R']-half),
                          default={'accepted': n_acc})['accepted']
                cert_keys.add(mid)
            for ci, ak in enumerate(sorted(cert_keys,
                                           reverse=True)[
                                           :PROTOCOL['certified_poses_per_seed']]):
                if (time.monotonic() >= seed_deadline or srec['solver_calls']
                        >= PROTOCOL['max_solver_calls_per_seed']):
                    srec['certifications'].append(
                        {'accepted_index': ak, 'status': 'budget_exhausted'})
                    continue
                q_cert = acc_arrays[f'acc_{ak:04d}_qpos']
                cp = dp+f'cert_{ci:02d}_'
                crec, c_arrays = _certify(
                    body, transmission, q_cert, fmax, l0, passive, padrs,
                    pstiff, pranges, seed_deadline, cp)
                crec['accepted_index'] = int(ak)
                crec['status'] = 'completed'
                arrays.update(c_arrays)
                srec['certifications'].append(crec)
                srec['solver_calls'] += crec['solver_calls']
            srec['status'] = 'completed'
            result['seeds'].append(srec)
        after = {p.relative_to(workspace).as_posix(): file_hash(p)
                 for p in inputs}
        if after != result['input_sha256']:
            raise ValueError('Input evidence changed during the descent run')
        completed = [s for s in result['seeds']
                     if s.get('status') == 'completed']
        truncated = (len(completed) < len(seeds)
                     or any(s.get('remaining_seeds_unexecuted')
                            for s in result['seeds'])
                     or any(c.get('status') != 'completed'
                            for s in completed for c in s['certifications']))
        certs = [c for s in completed for c in s['certifications']
                 if c.get('status') == 'completed']
        any_full = any(c.get('any_full_feasible') for c in certs)
        any_sub = any(c.get('any_passive_feasible') for c in certs)
        undecided = any(
            any(s.get('passive_subsystem', {}).get('status') not in
                ('feasible', INFEASIBLE)
                or s.get('support', {}).get('status') not in
                (FEASIBLE_POSE, INFEASIBLE)
                or s.get('margin', {}).get('status') not in
                ('measured', 'unreachable_at_any_shift')
                for s in c['samples'] if s.get('evidence_saved'))
            for c in certs)
        result.update(status='completed',
                      seeds_executed=len(completed),
                      certifications_executed=len(certs),
                      descent_calls=sum(s['solver_calls'] for s in completed))
        if truncated or undecided:
            result['scientific_outcome'] = 'inconclusive'
        elif any_full:
            result['scientific_outcome'] = 'found_full_support_pose'
        elif any_sub:
            result['scientific_outcome'] = 'found_subsystem_feasible_pose'
        else:
            result['scientific_outcome'] = 'descent_floored_without_feasibility'
    except TimeoutError:
        result.update(status='timeout', scientific_outcome='invalid_experiment')
    except Exception as exc:
        result.update(status='error', scientific_outcome='invalid_experiment',
                      error=f'{type(exc).__name__}: {exc}')
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(out/'result.json', result)
    np.savez_compressed(out/'observations.npz', **arrays)
    return result


def _check_descent_replay(si, srec, arrays, rows):
    """Replay every saved accepted-step LP; returns (undecided, errs)."""
    errs = []
    undecided = False
    dp = f'desc_{si:02d}_'
    T = arrays[dp+'seed_tendon_map']
    C = arrays[dp+'seed_contact_map']
    friction = arrays[dp+'seed_friction']
    limits = arrays[dp+'seed_limits']
    descent = srec['descent']
    accepted = descent['accepted']
    for a in accepted:
        ak = a['accepted']
        target = arrays[dp+f'acc_{ak:04d}_target']
        R_replay, st = _residual(T, C, target, limits, friction, rows)
        if st not in ('feasible', 'infeasible'):
            undecided = True
        if a['R'] is None:
            if np.isfinite(R_replay):
                errs.append(
                    f'accepted step {ak} replay finite but recorded non-finite')
            continue
        if abs(R_replay-a['R']) > PROTOCOL['replay_tolerance']*max(1., abs(a['R'])):
            errs.append(f'accepted step {ak} replay R differs')
    q_term = arrays[dp+f"acc_{descent['accepted_steps']:04d}_qpos"]
    if not np.allclose(q_term, descent['terminal_qpos'], rtol=0, atol=1e-12):
        errs.append('terminal qpos does not match the last accepted step')
    last = accepted[-1]['R']
    if (descent['final_R'] is None) != (last is None):
        errs.append('recorded final_R finiteness differs from last accepted R')
    elif (last is not None and abs(descent['final_R']-last)
            > 1e-12*max(1., abs(last))):
        errs.append('recorded final_R differs from last accepted R')
    return undecided, errs


def verify(workspace):
    """Replay the descent trajectory and every certification LP."""
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import read_json
    workspace = Path(workspace).resolve()
    d = workspace/'residual-descent'
    verdict = {'schema': 1, 'status': 'running', 'evidence_valid': False,
               'scientific_outcome': 'invalid_experiment', 'errors': []}

    def fail(msg):
        verdict['errors'].append(msg)
        verdict['status'] = 'failed'
        write_json(d/'verification.json', verdict)
        return verdict
    try:
        result = read_json(d/'result.json')
        proto = read_json(d/'protocol.json')
        if result.get('protocol_sha256') != protocol_hash():
            return fail('Protocol hash mismatch')
        if proto.get('protocol') != PROTOCOL:
            return fail('Frozen protocol differs from implementation')
        for rel, expected in (result.get('input_sha256') or {}).items():
            p = workspace/rel
            if not p.is_file() or file_hash(p) != expected:
                return fail(f'Input hash mismatch: {rel}')
        with np.load(d/'observations.npz', allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        from .static_support import solve_support
        undecided = False
        n_traj = n_wit = 0
        saved_prefixes = sorted({m.group(1) for k in arrays
                                 for m in [WITNESS.match(k)] if m})
        claimed = set()
        for si, srec in enumerate(result.get('seeds', [])):
            if srec.get('status') != 'completed':
                continue
            index = srec['meta']['index']
            pp = f'pose_{index:02d}_'
            with np.load(workspace/'passive-posture/observations.npz',
                         allow_pickle=False) as z:
                passive = [int(i) for i in z[pp+'passive_dofs']]
            rows = list(range(6))+passive
            und, errs = _check_descent_replay(si, srec, arrays, rows)
            undecided = undecided or und
            n_traj += len(srec['descent']['accepted'])
            for e in errs:
                verdict['errors'].append(f'seed {si}: {e}')
            crecs = [c for c in srec.get('certifications', [])
                     if c.get('status') == 'completed']
            for ci, crec in enumerate(crecs):
                for sample in crec['samples']:
                    n = sample['index']
                    prefix = f'desc_{si:02d}_cert_{ci:02d}_depth_{n:02d}_'
                    if sample.get('evidence_saved') is not True:
                        continue
                    claimed.add(prefix)
                    if prefix+'support_tendon_map' not in arrays:
                        verdict['errors'].append(
                            f'{prefix} marked evidence_saved but missing arrays')
                        continue
                    n_wit += 1
                    T = arrays[prefix+'support_tendon_map']
                    C = arrays[prefix+'support_contact_map']
                    target = arrays[prefix+'support_target']
                    limits = arrays[prefix+'support_limits']
                    friction = arrays[prefix+'support_friction']
                    support = solve_support(T, C, target, limits, friction)
                    if support['status'] not in ('feasible', INFEASIBLE):
                        undecided = True
                    rep_support = (FEASIBLE_POSE if support['status'] == 'feasible'
                                   else support['status'])
                    if rep_support != sample['support']['status']:
                        verdict['errors'].append(
                            f'{prefix} support replay differs '
                            f"({rep_support} != {sample['support']['status']})")
                    subsystem = solve_subsystem(T, C, target, limits, friction,
                                                rows)
                    if subsystem['status'] not in ('feasible', INFEASIBLE):
                        undecided = True
                    if subsystem['status'] != sample['passive_subsystem']['status']:
                        verdict['errors'].append(
                            f'{prefix} subsystem replay differs')
                    intervals = arrays[prefix+'margin_intervals']
                    margin = margin_solve(T, C, target, limits, friction,
                                          list(range(6)), passive, intervals)
                    if margin['status'] == 'measured':
                        mt = sample['margin'].get('t')
                        if (mt is None or abs(margin['t']-mt)
                                > PROTOCOL['replay_tolerance']*max(1., abs(mt))):
                            verdict['errors'].append(
                                f'{prefix} margin t replay differs')
                    elif sample['margin']['status'] != margin['status']:
                        verdict['errors'].append(
                            f'{prefix} margin status replay differs')
        extra = set(saved_prefixes)-claimed
        if extra:
            verdict['errors'].append(
                f'saved witnesses not linked to a recorded sample: '
                f'{sorted(extra)[:4]}')
        if verdict['errors']:
            return fail('Evidence validation: '+'; '.join(verdict['errors'][:8]))
        completed = [s for s in result['seeds']
                     if s.get('status') == 'completed']
        truncated = (len(completed) < result.get('declared_seeds', 0)
                     or any(s.get('remaining_seeds_unexecuted')
                            for s in result['seeds'])
                     or any(c.get('status') != 'completed'
                            for s in completed for c in s['certifications']))
        certs = [c for s in completed for c in s['certifications']
                 if c.get('status') == 'completed']
        any_full = any(c.get('any_full_feasible') for c in certs)
        any_sub = any(c.get('any_passive_feasible') for c in certs)
        if truncated or undecided:
            replayed = 'inconclusive'
        elif any_full:
            replayed = 'found_full_support_pose'
        elif any_sub:
            replayed = 'found_subsystem_feasible_pose'
        else:
            replayed = 'descent_floored_without_feasibility'
        verdict.update(status='completed', evidence_valid=True,
                       scientific_outcome=replayed,
                       recorded_outcome=result.get('scientific_outcome'),
                       trajectory_steps_replayed=n_traj,
                       witnesses_checked=n_wit,
                       seeds_checked=len(completed))
        write_json(d/'verification.json', verdict)
        return verdict
    except Exception as exc:
        verdict['errors'].append(f'{type(exc).__name__}: {exc}')
        verdict['status'] = 'error'
        write_json(d/'verification.json', verdict)
        return verdict


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args(argv)
    out = verify(args.workspace) if args.verify else run(args.workspace)
    print(json.dumps({k: out.get(k) for k in
                      ('status', 'scientific_outcome', 'evidence_valid',
                       'errors')}, indent=2, default=str))


if __name__ == '__main__':
    main()
