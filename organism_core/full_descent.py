"""Live-contact descent on the FULL-system support residual (FE-02).

Every prior descent optimized the passive-subsystem residual — the
tension-free rows that block exact feasibility. The physically
meaningful objective is the full 72-row support residual: the minimum
scaled balance error achievable with all declared inequality constraints
(tension bounds, friction diamond) satisfied. This module descends that
objective under live contact re-derivation over 66 joint DOFs plus three
bounded root-rotation axes.

A terminal or accepted pose reaching full residual <= the declared
support tolerance is recorded as a tolerance witness: a force assignment
satisfying every bound and the friction diamond exists with equality
error within solver resolution, even though the strict LP never returns
feasible. Phase 2 still certifies poses through the unchanged gate chain.
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
from .passive_posture import FIELDS, INFEASIBLE, FEASIBLE_POSE
from .passive_margin import margin_solve, passive_rows
from .residual_descent import _certify
from .root_descent import _tilt, ROT_AXES

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PROTOCOL = {
    'id': 'FE-02-full-descent-v1',
    'seed_margin_threshold': 1e-3,
    'max_seeds_total': 8,
    'descent_dofs': ('every non-root DOF row (actuated and passive) plus '
                     'three world-frame root-rotation axes; root x,y are '
                     'dead by translation symmetry and root z is re-derived '
                     'per probe'),
    'objective': ('FULL-system minimum scaled balance error (all 72 rows '
                  'including tendon rows within declared tension limits) '
                  'with LIVE contact re-derivation per probe; minimum over '
                  'the depth grid, no root-balance pre-gate'),
    'descent_rule': ('deterministic cyclic coordinate descent: joint DOFs in '
                     'row order then rotation axes x,y,z; rotations applied '
                     'as world-frame axis-angle premultiplications; probes '
                     'exceeding max_root_tilt_rad from the seed orientation '
                     'are rejected without evaluation'),
    'h_rel_initial': 0.25, 'h_rel_floor': 1e-4,
    'joint_limit_inset_rad': 1e-4,
    'root_tilt_nominal_range_rad': 0.6,
    'max_root_tilt_rad': 0.6,
    'support_residual_tolerance': 1e-6,
    'max_evals_per_seed': 4000,
    'certified_poses_per_seed': 2,
    'depths_native': [.001, .005, .01, .02, .04],
    'bracket_half_range_native': 2., 'bracket_iterations': 25,
    'root_solver_time_limit_s': 2.,
    'max_solver_calls_per_seed': 60000,
    'seed_wall_s': 350., 'worker_wall_s': 1500., 'memory_mib': 8192,
    'replay_tolerance': 1e-6,
    'outcomes': ['found_full_support_pose', 'found_subsystem_feasible_pose',
                 'tolerance_support_witness',
                 'descent_floored_without_feasibility', 'inconclusive',
                 'invalid_experiment'],
    'verification': ('replay of saved-input LPs in the same SciPy/HiGHS '
                     'family; no independent physics or biological check'),
    'scope': ('Static witness generation; a tolerance witness is a force '
              'assignment satisfying all declared inequality constraints '
              'with equality residual within solver resolution — not exact '
              'LP feasibility, not a standing or dynamic claim'),
}
WITNESS = re.compile(r'^(fdesc_\d+_cert_\d+_depth_\d+_)support_tendon_map$')


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _full_residual(report):
    """Map a solve_support report to (R, status) for descent bookkeeping."""
    if report['status'] == 'feasible':
        return 0., 'feasible'
    if report['status'] == 'residual_check_failed':
        r = report.get('maximum_scaled_residual')
        if r is not None and np.isfinite(r):
            return float(r), 'residual_check_failed'
        return float('inf'), 'unresolved'
    if report['status'] == INFEASIBLE:
        nb = report.get('nearest_balance') or {}
        r = nb.get('minimum_scaled_balance_error', np.inf)
        if np.isfinite(r):
            return float(r), 'infeasible'
        return float('inf'), 'unresolved'
    return float('inf'), report.get('status', 'unresolved')


def _probe_full(body, transmission, q, fmax, l0, deadline):
    """Evaluate the full-support residual at one pose with live contacts.

    For each depth: forward dynamics, contact detection, tendon limits,
    then the full 72-row support LP. The objective is the minimum scaled
    balance error across depths; +inf when no sample is eligible or the
    LP cannot decide. Saves nothing — callers record evidence.
    """
    import mujoco as mj
    from .pose_probe import first_contact_height
    from .support_diagnostics import contact_columns, evaluate_support
    from .static_support import solve_support
    model, data = body.m, body.d
    calls = {'solver': 0}
    height = first_contact_height(model, data, q, deadline=deadline)
    best = (float('inf'), 'no_eligible_sample', None)
    for n, depth in enumerate(PROTOCOL['depths_native']):
        data.qpos[:] = q; data.qpos[2] = height-depth
        data.qvel[:] = 0; data.qacc_warmstart[:] = 0
        data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        mj.mj_forward(model, data)
        if np.any(data.warning.number):
            raise ValueError('Solver warning in full-descent probe')
        C, mu, contacts, nonfoot = contact_columns(model, data)
        legs = [leg for leg in LEGS
                if any('/'+leg+'_tarsus' in c['geom'] for c in contacts)]
        if not contacts or nonfoot:
            continue
        limits = fmax*np.exp(-((data.ten_length/l0-1)/.5)**2)
        _, inputs = evaluate_support(body, transmission, limits)
        calls['solver'] += 1
        sup = solve_support(*(inputs[k] for k in FIELDS[:4]),
                            inputs['support_friction'])
        calls['solver'] += 1
        R, st = _full_residual(sup)
        if R < best[0] or best[2] is None:
            best = (R, st, {'index': n, 'depth_native': depth,
                            'foot_legs': legs, 'support_status': st,
                            'inputs': inputs,
                            'ten_length': data.ten_length.copy()})
        if R == 0.:
            break
    return best[0], best[1], best[2], calls


def _descend_full(body, transmission, fmax, l0, dof_rows, dof_adrs,
                  dof_ranges, q_seed, deadline, budget):
    """Coordinate descent over joint DOFs plus three root-rotation axes."""
    import mujoco as mj
    q = np.asarray(q_seed, dtype=float).copy()
    R, st, detail, calls = _probe_full(body, transmission, q, fmax, l0,
                                       deadline)
    budget[0] -= 1
    _r = lambda v: float(v) if np.isfinite(v) else None
    rec = {'solver_calls': calls['solver'], 'probes': 1, 'unresolved_evals': 0,
           'sweeps': [], 'status': 'pending', 'initial_R': _r(R),
           'initial_status': st}
    accepted = {}
    n_acc = 0

    def record(qq, det):
        nonlocal n_acc
        sp = f'acc_{n_acc:04d}_'
        accepted[sp+'qpos'] = qq.copy()
        for k in FIELDS:
            accepted[sp+k] = np.asarray(det['inputs'][k], dtype=float)
        accepted[sp+'sample_index'] = np.asarray(det['index'], dtype=int)

    if detail is not None:
        record(q, detail)
    acc_rec = [{'accepted': 0, 'dof_row': -1, 'direction': 0, 'R': _r(R)}]
    inset = PROTOCOL['joint_limit_inset_rad']
    tilt_range = PROTOCOL['root_tilt_nominal_range_rad']
    tilt_max = PROTOCOL['max_root_tilt_rad']
    h_rel = PROTOCOL['h_rel_initial']
    while budget[0] > 0 and h_rel >= PROTOCOL['h_rel_floor']:
        if time.monotonic() >= deadline:
            rec['status'] = 'wall_budget_exhausted'
            break
        sweep = {'h_rel': h_rel, 'improvements': 0}
        n_joint = len(dof_adrs)
        for di in range(n_joint+3):
            if budget[0] <= 0 or time.monotonic() >= deadline:
                break
            if di < n_joint:
                lo, hi = dof_ranges[di]
                h = h_rel*(hi-lo)
            else:
                h = h_rel*tilt_range
            best_R, best_q, best_det, best_sign = R, None, None, 0
            for sign in (1., -1.):
                if budget[0] <= 0:
                    break
                qp = q.copy()
                if di < n_joint:
                    adr = dof_adrs[di]
                    qp[adr] = min(max(q[adr]+sign*h, lo+inset), hi-inset)
                    if qp[adr] == q[adr]:
                        continue
                else:
                    axis = ROT_AXES[di-n_joint]
                    dq = np.zeros(4)
                    mj.mju_axisAngle2Quat(dq, axis, sign*h)
                    out = np.zeros(4)
                    mj.mju_mulQuat(out, dq, qp[3:7])
                    qp[3:7] = out/np.linalg.norm(out)
                    if _tilt(qp, np.asarray(q_seed)) > tilt_max:
                        continue
                try:
                    Rp, stp, detp, calls = _probe_full(body, transmission, qp,
                                                       fmax, l0, deadline)
                except ValueError as exc:
                    if 'not bracketed' in str(exc):
                        continue
                    raise
                budget[0] -= 1
                rec['solver_calls'] += calls['solver']; rec['probes'] += 1
                if stp not in ('feasible', 'infeasible',
                               'residual_check_failed'):
                    rec['unresolved_evals'] += 1
                if Rp < best_R:
                    best_R, best_q, best_det, best_sign = Rp, qp, detp, sign
            if best_q is not None:
                q, R, detail = best_q, best_R, best_det
                n_acc += 1
                record(q, detail)
                row = (int(dof_rows[di]) if di < n_joint
                       else -(di-n_joint+2))
                acc_rec.append({'accepted': n_acc, 'dof_row': row,
                                'direction': int(best_sign), 'R': _r(R)})
                sweep['improvements'] += 1
                if R == 0.:
                    break
        sweep['R_after'] = _r(R)
        rec['sweeps'].append(sweep)
        if R == 0.:
            rec['status'] = 'residual_zero_live'
            break
        if sweep['improvements'] == 0:
            h_rel *= .5
    if rec['status'] == 'pending':
        rec['status'] = ('eval_budget_exhausted' if budget[0] <= 0
                         else 'wall_budget_exhausted'
                         if time.monotonic() >= deadline
                         else 'step_floor_reached')
    rec.update(final_R=_r(R), accepted_steps=n_acc, accepted=acc_rec,
               terminal_qpos=q.tolist(),
               terminal_tilt_rad=float(_tilt(q, np.asarray(q_seed))),
               terminal_driver=(detail if detail is None else
                                {k: detail[k] for k in
                                 ('index', 'depth_native', 'foot_legs',
                                  'support_status')}))
    return rec, accepted, q, detail


def run(workspace):
    """Descend the live full-support residual, then certify terminal poses."""
    import mujoco as mj
    from .body import Body
    from .root_ab import derive_xml, audit_models, file_hash, write_json
    from .root_ab_evidence import read_json
    from .transmission import SiteTendonTransmission
    workspace = Path(workspace).resolve()
    out = workspace/'full-descent'; out.mkdir(exist_ok=False)
    placed_dir = workspace/'foot-placement'
    posture_dir = workspace/'passive-posture'
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
                       /'examples/fe02-full-descent.json')
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
                ('full_descent.py', 'root_descent.py', 'contact_descent.py',
                 'residual_descent.py', 'iterative_shift.py',
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
            srec = {'meta': seed, 'status': 'running', 'solver_calls': 0,
                    'certifications': []}
            budget = [PROTOCOL['max_evals_per_seed']]
            try:
                drec, acc_arrays, q_term, detail = _descend_full(
                    body, transmission, fmax, l0, dof_rows, dof_adrs,
                    dof_ranges, q_seed, seed_deadline, budget)
            except ValueError as exc:
                if 'not bracketed' in str(exc):
                    srec['status'] = 'not_bracketed'
                    srec['descent'] = {'status': 'not_bracketed'}
                    result['seeds'].append(srec)
                    continue
                raise
            srec['solver_calls'] += drec['solver_calls']
            srec['probes'] = drec['probes']
            srec['unresolved_evals'] = drec['unresolved_evals']
            srec['descent'] = {k: v for k, v in drec.items()
                               if k not in ('solver_calls', 'probes')}
            dp = f'fdesc_{si:02d}_'
            arrays.update({dp+k: v for k, v in acc_arrays.items()})
            arrays[dp+'seed_qpos'] = np.asarray(q_seed, dtype=float)
            n_acc = drec['accepted_steps']
            cert_keys = {n_acc}
            if (n_acc > 1 and drec['initial_R'] is not None
                    and drec['final_R'] is not None):
                half = (drec['initial_R']+drec['final_R'])/2.
                mid = min((a for a in drec['accepted'] if a['R'] is not None),
                          key=lambda a: abs(a['R']-half),
                          default={'accepted': n_acc})['accepted']
                cert_keys.add(mid)
            for ci, ak in enumerate(sorted(cert_keys, reverse=True)[
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
        tol = PROTOCOL['support_residual_tolerance']
        tol_witness = any(
            a['R'] is not None and a['R'] <= tol
            for s in completed for a in s['descent']['accepted'])
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
                      descent_calls=sum(s['solver_calls'] for s in completed),
                      tolerance_witness=tol_witness)
        if truncated or undecided:
            result['scientific_outcome'] = 'inconclusive'
        elif any_full:
            result['scientific_outcome'] = 'found_full_support_pose'
        elif any_sub:
            result['scientific_outcome'] = 'found_subsystem_feasible_pose'
        elif tol_witness:
            result['scientific_outcome'] = 'tolerance_support_witness'
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


def _check_descent_replay(si, srec, arrays):
    """Replay every saved accepted-step full-support LP."""
    from .static_support import solve_support
    errs = []
    undecided = False
    dp = f'fdesc_{si:02d}_'
    descent = srec['descent']
    accepted = descent['accepted']
    for a in accepted:
        ak = a['accepted']
        sp = dp+f'acc_{ak:04d}_'
        if sp+'support_tendon_map' not in arrays:
            if a['R'] is None:
                continue
            errs.append(
                f'accepted step {ak} recorded finite R but saved no inputs')
            continue
        sup = solve_support(arrays[sp+'support_tendon_map'],
                            arrays[sp+'support_contact_map'],
                            arrays[sp+'support_target'],
                            arrays[sp+'support_limits'],
                            arrays[sp+'support_friction'])
        R_replay, st = _full_residual(sup)
        if st not in ('feasible', 'infeasible', 'residual_check_failed'):
            undecided = True
        if a['R'] is None:
            if np.isfinite(R_replay):
                errs.append(
                    f'accepted step {ak} replay finite but recorded non-finite')
            continue
        if abs(R_replay-a['R']) > PROTOCOL['replay_tolerance']*max(1., abs(a['R'])):
            errs.append(f'accepted step {ak} replay R differs')
    last_key = descent['accepted_steps']
    if f'acc_{last_key:04d}_qpos' in arrays:
        q_term = arrays[dp+f'acc_{last_key:04d}_qpos']
        if not np.allclose(q_term, descent['terminal_qpos'], rtol=0, atol=1e-12):
            errs.append('terminal qpos does not match the last accepted step')
    last = accepted[-1]['R']
    if (descent['final_R'] is None) != (last is None):
        errs.append('recorded final_R finiteness differs from last accepted R')
    elif (last is not None and abs(descent['final_R']-last)
            > 1e-12*max(1., abs(last))):
        errs.append('recorded final_R differs from last accepted R')
    tilt = descent.get('terminal_tilt_rad')
    if tilt is not None and not np.isfinite(tilt):
        errs.append('terminal tilt is not finite')
    elif (tilt is not None
            and tilt > PROTOCOL['max_root_tilt_rad']+1e-9):
        errs.append('terminal tilt exceeds the declared bound')
    return undecided, errs


def verify(workspace):
    """Replay the full-descent trajectory and every certification LP."""
    from .passive_posture import solve_subsystem
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import read_json
    from .static_support import solve_support
    workspace = Path(workspace).resolve()
    d = workspace/'full-descent'
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
        undecided = False
        n_traj = n_wit = 0
        saved_prefixes = sorted({m.group(1) for k in arrays
                                 for m in [WITNESS.match(k)] if m})
        claimed = set()
        tol = PROTOCOL['support_residual_tolerance']
        tol_witness = False
        for si, srec in enumerate(result.get('seeds', [])):
            if srec.get('status') != 'completed':
                continue
            index = srec['meta']['index']
            pp = f'pose_{index:02d}_'
            with np.load(workspace/'passive-posture/observations.npz',
                         allow_pickle=False) as z:
                passive = [int(i) for i in z[pp+'passive_dofs']]
            rows = list(range(6))+passive
            und, errs = _check_descent_replay(si, srec, arrays)
            undecided = undecided or und
            n_traj += len(srec['descent']['accepted'])
            for a in srec['descent']['accepted']:
                if a['R'] is not None and a['R'] <= tol:
                    tol_witness = True
            for e in errs:
                verdict['errors'].append(f'seed {si}: {e}')
            crecs = [c for c in srec.get('certifications', [])
                     if c.get('status') == 'completed']
            for ci, crec in enumerate(crecs):
                for sample in crec['samples']:
                    n = sample['index']
                    prefix = f'fdesc_{si:02d}_cert_{ci:02d}_depth_{n:02d}_'
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
        elif tol_witness:
            replayed = 'tolerance_support_witness'
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
