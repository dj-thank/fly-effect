"""Bounded global posture sampling against the full-support residual (FE-02).

All previous descents explored the seed basin reached by the
iterative-shift lineages. This module asks whether the nonzero residual
floor is a local-basin artifact or a global feature of the posture
space: sample declared-random poses (joints uniform inside their limits,
root tilt uniform-axis/uniform-angle within a bounded ball, root x,y
fixed by translation symmetry, root z re-derived per probe), evaluate
the full 72-row support residual under live contacts, then run the same
bounded coordinate descent from the best eligible samples and certify
terminal poses through the unchanged gate chain.

Static witness generation only — no dynamic hold, standing, walking,
flight, CNS execution or biological validation. A bounded declared
sample is not global optimization; a negative means nothing sampled
beats the seed-basin floor, within the declared budget.
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
from .passive_margin import margin_solve, passive_rows
from .residual_descent import _certify
from .full_descent import _probe_full, _descend_full, _full_residual

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PROTOCOL = {
    'id': 'FE-02-global-sample-v1',
    'rng_seed': 20260915,
    'n_samples': 512,
    'sampling': ('per-hinge uniform inside joint limits (inset by '
                 'joint_limit_inset_rad); root orientation = uniform axis '
                 'on the sphere with uniform angle in [0, '
                 'max_root_tilt_rad] composed onto the neutral quat; '
                 'root x,y fixed (translation symmetry), root z re-derived '
                 'per probe by first-contact bracketing'),
    'objective': ('FULL-system minimum scaled balance error (all 72 rows '
                  'within declared tension limits) with LIVE contact '
                  're-derivation per probe; identical to '
                  'FE-02-full-descent-v1'),
    'descend_top_k': 8,
    'descent_evals_per_sample': 500,
    'descent_dofs': ('every non-root DOF row plus three world-frame '
                     'root-rotation axes, identical to FE-02-full-descent-v1'),
    'improvement_rule': ('a descended sample terminal R strictly below the '
                         'best recorded full-descent terminal residual '
                         '(read from the verified full-descent result)'),
    'h_rel_initial': 0.25, 'h_rel_floor': 1e-4,
    'joint_limit_inset_rad': 1e-4,
    'root_tilt_nominal_range_rad': 0.6,
    'max_root_tilt_rad': 0.6,
    'support_residual_tolerance': 1e-6,
    'certified_poses_per_sample': 2,
    'depths_native': [.001, .005, .01, .02, .04],
    'bracket_half_range_native': 2., 'bracket_iterations': 25,
    'root_solver_time_limit_s': 2.,
    'max_solver_calls': 120000,
    'seed_wall_s': 350., 'worker_wall_s': 1500., 'memory_mib': 8192,
    'replay_tolerance': 1e-6,
    'outcomes': ['found_full_support_pose', 'found_subsystem_feasible_pose',
                 'tolerance_support_witness', 'global_basin_improvement',
                 'no_global_improvement', 'inconclusive',
                 'invalid_experiment'],
    'verification': ('replay of saved-input LPs in the same SciPy/HiGHS '
                     'family; no independent physics or biological check'),
    'scope': ('Static witness generation; a bounded declared sample is not '
              'global optimization; no dynamic or biological claim'),
}
WITNESS = re.compile(r'^(gdesc_\d+_cert_\d+_depth_\d+_)support_tendon_map$')
SAMPLE_WITNESS = re.compile(r'^(gsamp_\d+_)support_tendon_map$')


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _sample_qpos(rng, q0, dof_adrs, dof_ranges, model):
    """One declared-random pose: uniform hinges, bounded random tilt."""
    import mujoco as mj
    q = np.asarray(q0, dtype=float).copy()
    inset = PROTOCOL['joint_limit_inset_rad']
    for adr, (lo, hi) in zip(dof_adrs, dof_ranges):
        q[adr] = rng.uniform(lo+inset, hi-inset)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0., PROTOCOL['max_root_tilt_rad'])
    dq = np.zeros(4)
    mj.mju_axisAngle2Quat(dq, axis, angle)
    out = np.zeros(4)
    mj.mju_mulQuat(out, dq, q[3:7])
    q[3:7] = out/np.linalg.norm(out)
    return q


def run(workspace):
    """Sample global poses, descend the best, certify the terminals."""
    import mujoco as mj
    from .body import Body
    from .root_ab import derive_xml, audit_models, file_hash, write_json
    from .root_ab_evidence import read_json
    from .transmission import SiteTendonTransmission
    workspace = Path(workspace).resolve()
    out = workspace/'global-sample'; out.mkdir(exist_ok=False)
    placed_dir = workspace/'foot-placement'
    posture_dir = workspace/'passive-posture'
    shift_dir = workspace/'iterative-shift'
    full_dir = workspace/'full-descent'
    source_dir = workspace/'source-study'
    started = time.monotonic(); deadline = started+PROTOCOL['worker_wall_s']
    result = {'schema': 1, 'protocol': PROTOCOL,
              'protocol_sha256': protocol_hash(), 'status': 'running',
              'scientific_outcome': 'not_run', 'samples': [],
              'biological_validation': False, 'walking_claimed': False,
              'standing_claimed': False, 'CNS_executed': False,
              'dynamic_trials_executed': 0, 'new_physics_trajectories': 0}
    arrays = {}
    try:
        record_path = (Path(__file__).resolve().parents[1]
                       /'examples/fe02-global-sample.json')
        if read_json(record_path)['parameters'] != PROTOCOL:
            raise ValueError('Versioned record differs from implemented protocol')
        full_verdict = read_json(full_dir/'verification.json')
        if full_verdict.get('evidence_valid') is not True:
            raise ValueError('Full-descent evidence was not verified')
        full_result = read_json(full_dir/'result.json')
        seed_floor = min(
            s['descent']['final_R'] for s in full_result['seeds']
            if s.get('descent', {}).get('final_R') is not None)
        inputs = [placed_dir/'result.json', placed_dir/'observations.npz',
                  placed_dir/'model-receipt.json', workspace/'protocol.json',
                  source_dir/'result.json', source_dir/'observations.npz',
                  posture_dir/'result.json', posture_dir/'observations.npz',
                  shift_dir/'result.json', shift_dir/'observations.npz',
                  shift_dir/'verification.json',
                  full_dir/'result.json', full_dir/'observations.npz',
                  full_dir/'verification.json']
        record_inputs = {p.relative_to(workspace).as_posix(): file_hash(p)
                         for p in inputs}
        code = {name: file_hash(Path(__file__).with_name(name)) for name in
                ('global_sample.py', 'full_descent.py', 'root_descent.py',
                 'contact_descent.py', 'residual_descent.py',
                 'iterative_shift.py', 'passive_margin.py',
                 'passive_posture.py', 'static_support.py',
                 'foot_placement.py', 'support_diagnostics.py',
                 'pose_probe.py', 'root_ab_evidence.py')}
        frozen = {'schema': 1, 'protocol': PROTOCOL,
                  'protocol_sha256': protocol_hash(),
                  'input_sha256': record_inputs, 'code_sha256': code,
                  'versions': {n: importlib.metadata.version(n)
                               for n in ('numpy', 'scipy', 'mujoco', 'flygym')}}
        write_json(out/'protocol.json', frozen)
        source = read_json(source_dir/'result.json')
        placed = read_json(placed_dir/'result.json')
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
                      versions=frozen['versions'], units=source['units'],
                      baseline_floor=seed_floor)
        body = Body.__new__(Body); body.m = model; body.d = mj.MjData(model)
        data = body.d; transmission = SiteTendonTransmission(model)
        with np.load(source_dir/'observations.npz', allow_pickle=False) as z:
            q0 = z['neutral_qpos'].copy()
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
        rng = np.random.default_rng(PROTOCOL['rng_seed'])
        budget = [PROTOCOL['max_solver_calls']]
        samples = []
        for i in range(PROTOCOL['n_samples']):
            if time.monotonic() >= deadline or budget[0] <= 0:
                result['samples'].append(
                    {'index': i, 'status': 'not_run',
                     'remaining_samples_unexecuted': True})
                break
            q = _sample_qpos(rng, q0, dof_adrs, dof_ranges, model)
            srec = {'index': i, 'status': 'running'}
            try:
                R, st, det, calls = _probe_full(body, transmission, q,
                                                fmax, l0, deadline)
            except ValueError as exc:
                if 'not bracketed' in str(exc):
                    srec.update(status='not_bracketed', R=None)
                    result['samples'].append(srec)
                    continue
                raise
            budget[0] -= calls['solver']
            eligible = (det is not None and np.isfinite(R))
            srec.update(status='evaluated', R=(float(R) if np.isfinite(R)
                                               else None),
                        probe_status=st, eligible=bool(eligible),
                        solver_calls=calls['solver'])
            if det is not None:
                srec['foot_legs'] = det['foot_legs']
                sp = f'gsamp_{i:03d}_'
                arrays[sp+'qpos'] = q.copy()
                for k in FIELDS:
                    arrays[sp+k] = np.asarray(det['inputs'][k], dtype=float)
                arrays[sp+'sample_index'] = np.asarray(det['index'], dtype=int)
                srec['evidence_saved'] = True
            else:
                srec['evidence_saved'] = False
            samples.append({'i': i, 'R': R, 'q': q, 'eligible': eligible})
            result['samples'].append(srec)
        eligible_sorted = sorted((s for s in samples if s['eligible']),
                                 key=lambda s: s['R'])
        top = eligible_sorted[:PROTOCOL['descend_top_k']]
        result['eligible_samples'] = len(eligible_sorted)
        for si, samp in enumerate(top):
            if time.monotonic() >= deadline or budget[0] <= 0:
                result['samples'][samp['i']]['descent'] = {
                    'status': 'budget_exhausted'}
                continue
            dbudget = [PROTOCOL['descent_evals_per_sample']]
            try:
                drec, acc_arrays, q_term, detail = _descend_full(
                    body, transmission, fmax, l0, dof_rows, dof_adrs,
                    dof_ranges, samp['q'], deadline, dbudget)
            except ValueError as exc:
                if 'not bracketed' in str(exc):
                    result['samples'][samp['i']]['descent'] = {
                        'status': 'not_bracketed'}
                    continue
                raise
            budget[0] -= drec['solver_calls']
            dp = f'gdesc_{si:02d}_'
            arrays.update({dp+k: v for k, v in acc_arrays.items()})
            arrays[dp+'sample_qpos'] = np.asarray(samp['q'], dtype=float)
            n_acc = drec['accepted_steps']
            cert_keys = {n_acc}
            if (n_acc > 1 and drec['initial_R'] is not None
                    and drec['final_R'] is not None):
                half = (drec['initial_R']+drec['final_R'])/2.
                mid = min((a for a in drec['accepted'] if a['R'] is not None),
                          key=lambda a: abs(a['R']-half),
                          default={'accepted': n_acc})['accepted']
                cert_keys.add(mid)
            certs = []
            for ci, ak in enumerate(sorted(cert_keys, reverse=True)[
                    :PROTOCOL['certified_poses_per_sample']]):
                if time.monotonic() >= deadline or budget[0] <= 0:
                    certs.append({'accepted_index': ak,
                                  'status': 'budget_exhausted'})
                    continue
                q_cert = acc_arrays[f'acc_{ak:04d}_qpos']
                sp = dp+f'acc_{ak:04d}_'
                T_saved = arrays[sp+'support_tendon_map']
                passive = passive_rows(T_saved, list(range(6)))
                joint_ids = np.array([int(model.dof_jntid[d])
                                      for d in passive])
                padrs = model.jnt_qposadr[joint_ids]
                pranges = model.jnt_range[joint_ids]
                pstiff = model.jnt_stiffness[joint_ids]
                cp = dp+f'cert_{ci:02d}_'
                crec, c_arrays = _certify(
                    body, transmission, q_cert, fmax, l0, passive, padrs,
                    pstiff, pranges, deadline, cp)
                crec['accepted_index'] = int(ak)
                crec['status'] = 'completed'
                arrays.update(c_arrays)
                budget[0] -= crec['solver_calls']
                certs.append(crec)
            drec_out = {k: v for k, v in drec.items()
                        if k not in ('solver_calls', 'probes')}
            drec_out['certifications'] = certs
            result['samples'][samp['i']]['descent'] = drec_out
            result['samples'][samp['i']]['descended'] = True
            result['samples'][samp['i']]['descent_ordinal'] = si
        after = {p.relative_to(workspace).as_posix(): file_hash(p)
                 for p in inputs}
        if after != result['input_sha256']:
            raise ValueError('Input evidence changed during the sampling run')
        evaluated = [s for s in result['samples']
                     if s.get('status') == 'evaluated']
        truncated = (len(evaluated) < PROTOCOL['n_samples']
                     or any(s.get('remaining_samples_unexecuted')
                            for s in result['samples']))
        descended = [s for s in result['samples']
                     if s.get('descent', {}).get('status')
                     not in (None, 'budget_exhausted', 'not_bracketed')]
        certs = [c for s in descended
                 for c in s['descent'].get('certifications', [])
                 if c.get('status') == 'completed']
        cert_trunc = any(c.get('status') != 'completed'
                         for s in descended
                         for c in s['descent'].get('certifications', []))
        any_full = any(c.get('any_full_feasible') for c in certs)
        any_sub = any(c.get('any_passive_feasible') for c in certs)
        tol = PROTOCOL['support_residual_tolerance']
        tol_witness = any(
            a['R'] is not None and a['R'] <= tol
            for s in descended for a in s['descent']['accepted'])
        tol_witness = tol_witness or any(
            s.get('R') is not None and s['R'] <= tol for s in evaluated)
        improved = any(
            s['descent'].get('final_R') is not None
            and s['descent']['final_R'] < seed_floor
            for s in descended)
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
                      samples_evaluated=len(evaluated),
                      samples_descended=len(descended),
                      certifications_executed=len(certs),
                      descent_calls=PROTOCOL['max_solver_calls']-budget[0],
                      tolerance_witness=tol_witness)
        if truncated or cert_trunc or undecided:
            result['scientific_outcome'] = 'inconclusive'
        elif any_full:
            result['scientific_outcome'] = 'found_full_support_pose'
        elif any_sub:
            result['scientific_outcome'] = 'found_subsystem_feasible_pose'
        elif tol_witness:
            result['scientific_outcome'] = 'tolerance_support_witness'
        elif improved:
            result['scientific_outcome'] = 'global_basin_improvement'
        else:
            result['scientific_outcome'] = 'no_global_improvement'
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
    """Replay every saved accepted-step full-support LP for one sample."""
    from .static_support import solve_support
    errs = []
    undecided = False
    dp = f'gdesc_{si:02d}_'
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
    """Replay every saved sample, descent step and certification LP."""
    from .root_ab import file_hash, write_json
    from .root_ab_evidence import read_json
    from .static_support import solve_support
    workspace = Path(workspace).resolve()
    d = workspace/'global-sample'
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
        sample_prefixes = {m.group(1) for k in arrays
                           for m in [SAMPLE_WITNESS.match(k)] if m}
        claimed_samples = set()
        tol = PROTOCOL['support_residual_tolerance']
        tol_witness = False
        for srec in result.get('samples', []):
            if srec.get('status') != 'evaluated':
                continue
            i = srec['index']
            sp = f'gsamp_{i:03d}_'
            if srec.get('evidence_saved'):
                claimed_samples.add(sp)
                if sp+'support_tendon_map' not in arrays:
                    verdict['errors'].append(
                        f'{sp} marked evidence_saved but missing arrays')
                    continue
                n_wit += 1
                sup = solve_support(arrays[sp+'support_tendon_map'],
                                    arrays[sp+'support_contact_map'],
                                    arrays[sp+'support_target'],
                                    arrays[sp+'support_limits'],
                                    arrays[sp+'support_friction'])
                R_replay, st = _full_residual(sup)
                if st not in ('feasible', 'infeasible',
                              'residual_check_failed'):
                    undecided = True
                if srec['R'] is None:
                    if np.isfinite(R_replay):
                        verdict['errors'].append(
                            f'{sp} replay finite but recorded non-finite')
                elif abs(R_replay-srec['R']) > tol*max(1., abs(srec['R'])):
                    verdict['errors'].append(f'{sp} replay R differs')
                if srec['R'] is not None and srec['R'] <= tol:
                    tol_witness = True
        extra_samples = sample_prefixes-claimed_samples
        if extra_samples:
            verdict['errors'].append(
                f'saved sample witnesses unlinked: '
                f'{sorted(extra_samples)[:4]}')
        claimed = set()
        for srec in (s for s in result.get('samples', [])
                     if s.get('descent', {}).get('status')
                     not in (None, 'budget_exhausted', 'not_bracketed')):
            si = srec.get('descent_ordinal')
            if si is None:
                verdict['errors'].append(
                    f"sample {srec['index']} descended but no ordinal recorded")
                continue
            und, errs = _check_descent_replay(si, srec, arrays)
            undecided = undecided or und
            n_traj += len(srec['descent']['accepted'])
            for a in srec['descent']['accepted']:
                if a['R'] is not None and a['R'] <= tol:
                    tol_witness = True
            for e in errs:
                verdict['errors'].append(f'sample {si}: {e}')
            for ci, crec in enumerate(
                    c for c in srec['descent'].get('certifications', [])
                    if c.get('status') == 'completed'):
                for sample in crec['samples']:
                    n = sample['index']
                    prefix = f'gdesc_{si:02d}_cert_{ci:02d}_depth_{n:02d}_'
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
                    passive = sample['passive_rows']
                    rows = list(range(6))+passive
                    support = solve_support(T, C, target, limits, friction)
                    if support['status'] not in ('feasible', INFEASIBLE):
                        undecided = True
                    rep_support = (FEASIBLE_POSE if support['status'] == 'feasible'
                                   else support['status'])
                    if rep_support != sample['support']['status']:
                        verdict['errors'].append(
                            f'{prefix} support replay differs')
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
                f'saved cert witnesses not linked: {sorted(extra)[:4]}')
        if verdict['errors']:
            return fail('Evidence validation: '+'; '.join(verdict['errors'][:8]))
        evaluated = [s for s in result['samples']
                     if s.get('status') == 'evaluated']
        truncated = (len(evaluated) < PROTOCOL['n_samples']
                     or any(s.get('remaining_samples_unexecuted')
                            for s in result['samples']))
        descended = [s for s in result['samples']
                     if s.get('descent', {}).get('status')
                     not in (None, 'budget_exhausted', 'not_bracketed')]
        certs = [c for s in descended
                 for c in s['descent'].get('certifications', [])
                 if c.get('status') == 'completed']
        cert_trunc = any(c.get('status') != 'completed'
                         for s in descended
                         for c in s['descent'].get('certifications', []))
        any_full = any(c.get('any_full_feasible') for c in certs)
        any_sub = any(c.get('any_passive_feasible') for c in certs)
        improved = any(
            s['descent'].get('final_R') is not None
            and s['descent']['final_R'] < result['baseline_floor']
            for s in descended)
        if truncated or cert_trunc or undecided:
            replayed = 'inconclusive'
        elif any_full:
            replayed = 'found_full_support_pose'
        elif any_sub:
            replayed = 'found_subsystem_feasible_pose'
        elif tol_witness:
            replayed = 'tolerance_support_witness'
        elif improved:
            replayed = 'global_basin_improvement'
        else:
            replayed = 'no_global_improvement'
        verdict.update(status='completed', evidence_valid=True,
                       scientific_outcome=replayed,
                       recorded_outcome=result.get('scientific_outcome'),
                       trajectory_steps_replayed=n_traj,
                       witnesses_checked=n_wit,
                       samples_checked=len(evaluated))
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
