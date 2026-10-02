"""Bounded diagnostic foot placements; never a controller or gait certificate.

Only initial hinge coordinates change. The free root is fixed during IK; a
single vertical initial placement then brackets ground contact. No simulation
step is root-forced. All model parameters and support tolerances are preserved.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PROTOCOL = {
    'id': 'FE-02-foot-placement-v1', 'seed': 0,
    'candidate_count': 10, 'xy_scales': [.8, 1., 1.2],
    'target_z_offsets_native': [0., .15, .3],
    'depths_native': [.001, .005, .01, .02, .04],
    'ik_max_nfev': 80, 'ik_regularization': .001,
    'ik_tolerance': 1e-8, 'joint_limit_inset_rad': 1e-4,
    'root_solver_time_limit_s': 2., 'worker_wall_s': 240.,
    'outer_wall_s': 600., 'memory_mib': 8192,
    'selection': 'root feasible first, then most distinct feet, then shallowest depth',
    'targets': 'neutral terminal tarsus5 body origins; scaled XY around root; min neutral Z plus offset',
    'full_support': 'one selected root-feasible depth per candidate, original solver/force bounds',
    'dynamic_trials': 0, 'biological_validation': False,
}


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _deadline(deadline):
    if not np.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Foot-placement wall budget exceeded')


def root_balance(contact_map, target, friction):
    """Necessary contact-only balance at six free-root DOFs; not muscle support."""
    from scipy.optimize import linprog
    C, b, mu = (np.asarray(v, dtype=float) for v in (contact_map, target, friction))
    if (C.ndim != 2 or C.shape[0] != 6 or C.shape[1] == 0 or C.shape[1] % 3
            or b.shape != (6,) or mu.shape != (C.shape[1]//3,)):
        raise ValueError('Root balance requires six rows and three columns per contact')
    if not all(np.isfinite(v).all() for v in (C, b, mu)) or np.any(mu < 0):
        raise ValueError('Finite root balance data and nonnegative friction required')
    k = len(mu); inequalities = np.zeros((4*k, 3*k))
    for j in range(k):
        for r, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            inequalities[4*j+r, 3*j:3*j+3] = [sx, sy, -mu[j]]
    scale = np.maximum(np.abs(b), 1e-6)
    solved = linprog(np.zeros(3*k), A_eq=C/scale[:, None], b_eq=b/scale,
                     A_ub=inequalities, b_ub=np.zeros(4*k),
                     bounds=[(None, None), (None, None), (0., None)]*k,
                     method='highs', options={'time_limit': PROTOCOL['root_solver_time_limit_s']})
    result = {'feasible': False, 'status': 'inconclusive', 'solver_status': int(solved.status),
              'scope': 'Necessary free-root contact balance only; not full muscle feasibility'}
    if solved.success:
        x = np.asarray(solved.x, dtype=float)
        if x.shape != (3*k,) or not np.isfinite(x).all():
            result['status'] = 'invalid_solver_solution'
            return result
        error = float(np.max(np.abs(C@x-b)/scale))
        violation = float(max(0., np.max(inequalities@x), np.max(-x[2::3])))
        result.update(feasible=error <= 1e-6 and violation <= 1e-6,
                      forces_native=x.reshape(k, 3).tolist(), maximum_scaled_residual=error,
                      maximum_constraint_violation=violation)
        result['status'] = 'feasible' if result['feasible'] else 'residual_check_failed'
    elif solved.status == 2:
        result['status'] = 'infeasible_under_declared_constraints'
    return result


def fit_foot_origins(model, q0, joint_ids, body_ids, targets, *, deadline):
    """Bounded IK with analytic Jacobians and cached kinematics, on scratch data."""
    import mujoco as mj
    from scipy.optimize import least_squares
    _deadline(deadline)
    q0 = np.asarray(q0, dtype=float)
    ids = np.asarray(joint_ids, dtype=int); bodies = np.asarray(body_ids, dtype=int)
    targets = np.asarray(targets, dtype=float)
    if (q0.shape != (model.nq,) or ids.ndim != 1 or not ids.size
            or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= model.njnt)
            or bodies.ndim != 1 or not bodies.size or np.any(bodies <= 0) or np.any(bodies >= model.nbody)
            or targets.shape != (len(bodies), 3) or not np.isfinite(q0).all() or not np.isfinite(targets).all()):
        raise ValueError('Invalid IK coordinates, joints or foot targets')
    if np.any(model.jnt_type[ids] != mj.mjtJoint.mjJNT_HINGE) or not np.all(model.jnt_limited[ids]):
        raise ValueError('IK can change only limited hinge joints, never the free root')
    qadr, dofs = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    inset = PROTOCOL['joint_limit_inset_rad']
    lower, upper = model.jnt_range[ids, 0]+inset, model.jnt_range[ids, 1]-inset
    if np.any(lower >= upper):
        raise ValueError('Empty IK joint interval')
    start = np.clip(q0[qadr], lower, upper)
    data = mj.MjData(model); cached = None; residual = jacobian = None; evaluations = 0
    regularization = PROTOCOL['ik_regularization']
    def evaluate(x):
        nonlocal cached, residual, jacobian, evaluations
        _deadline(deadline)
        if cached is not None and np.array_equal(cached, x):
            return
        data.qpos[:] = q0; data.qpos[qadr] = x
        # Jacobians need kinematics and COM positions, not contact solving/dynamics.
        mj.mj_kinematics(model, data); mj.mj_comPos(model, data)
        blocks = []
        for body in bodies:
            jac = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jac, None, int(body))
            blocks.append(jac[:, dofs])
        residual = np.r_[(data.xpos[bodies]-targets).ravel(), regularization*(x-start)]
        jacobian = np.vstack([*blocks, regularization*np.eye(len(ids))])
        cached = x.copy(); evaluations += 1
        if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
            raise ValueError('Nonfinite IK kinematics')
    def fun(x):
        evaluate(x); return residual.copy()
    def jac(x):
        evaluate(x); return jacobian.copy()
    tolerance = PROTOCOL['ik_tolerance']
    solved = least_squares(fun, start, jac=jac, bounds=(lower, upper), method='trf',
                           max_nfev=PROTOCOL['ik_max_nfev'], ftol=tolerance,
                           xtol=tolerance, gtol=tolerance)
    _deadline(deadline)
    q = q0.copy(); q[qadr] = solved.x
    evaluate(solved.x)
    return q, {'optimizer_success': bool(solved.success), 'optimizer_status': int(solved.status),
               'nfev': int(solved.nfev), 'kinematics_evaluations': evaluations,
               'maximum_target_error_native': float(np.max(np.linalg.norm(residual[:3*len(bodies)].reshape(-1, 3), axis=1))),
               'root_unchanged_during_ik': bool(np.array_equal(q[:7], q0[:7])),
               'within_original_joint_limits': bool(np.all(solved.x >= lower) and np.all(solved.x <= upper)),
               'artificial_initial_placement': True, 'walking_claimed': False}


def select_sample(samples):
    eligible = [s for s in samples if s['foot_legs'] and not s['nonfoot_contacts']]
    if not eligible:
        return None
    return min(eligible, key=lambda s: (not s['root_balance']['feasible'], -len(s['foot_legs']), s['depth_native']))



def validate_sample(sample, contact_map, friction):
    """Cross-check contact labels/columns and any claimed force witness."""
    C, mu = np.asarray(contact_map, dtype=float), np.asarray(friction, dtype=float)
    contacts = sample['contacts']
    if C.shape != (6, 3*len(contacts)) or mu.shape != (len(contacts),):
        raise ValueError('Contact count/column mismatch')
    feet = [leg for leg in LEGS if any('/'+leg+'_tarsus' in c['geom'] for c in contacts)]
    if feet != sample['foot_legs'] or (contacts and not feet):
        raise ValueError('Contact foot labels disagree')
    if not np.array_equal(mu, [c['mu'] for c in contacts]):
        raise ValueError('Contact friction metadata mismatch')
    for contact in contacts:
        if contact['efc_address'] < 0 or not any('/'+leg+'_tarsus' in contact['geom'] for leg in LEGS):
            raise ValueError('Inactive/nonfoot contact column')
    eligible = bool(contacts) and not sample['nonfoot_contacts']
    if not eligible and sample['root_balance'] != {'feasible': False, 'status': 'not_eligible'}:
        raise ValueError('Ineligible contact sample makes a balance claim')


def root_contact_map(qpos, contacts):
    """Reconstruct free-root columns from saved world contact positions.

    Translation is world-frame; angular free-joint velocities use the root frame.
    This is a saved-coordinate Newton-Euler check, not a second collision engine.
    """
    q = np.asarray(qpos, dtype=float)
    if (q.ndim != 1 or len(q) < 7 or not np.isfinite(q).all()
            or not np.isclose(np.linalg.norm(q[3:7]), 1., atol=1e-10, rtol=0.)):
        raise ValueError('Finite coordinates and a unit root quaternion required')
    w, x, y, z = q[3:7]
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    blocks = []
    for contact in contacts:
        point = np.asarray(contact['position_native'], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Finite world contact position required')
        x, y, z = point-q[:3]
        moment = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        blocks.append(np.vstack([np.eye(3), rotation.T@moment]))
    return np.column_stack(blocks) if blocks else np.zeros((6, 0))


def validate_root_result(report, contact_map, target, friction):
    """Validate the retained force witness, not just a new LP's success flag."""
    C, b, mu = (np.asarray(v, dtype=float) for v in (contact_map, target, friction))
    if (mu.ndim != 1 or not len(mu) or C.shape != (6, 3*len(mu)) or b.shape != (6,)
            or not all(np.isfinite(v).all() for v in (C, b, mu)) or np.any(mu < 0)):
        raise ValueError('Invalid saved root balance inputs')
    status = report.get('status')
    if (type(report.get('feasible')) is not bool
            or report['feasible'] != (status == 'feasible')
            or type(report.get('solver_status')) is not int
            or status not in ('feasible', 'infeasible_under_declared_constraints', 'inconclusive')):
        raise ValueError('Root balance status/flag mismatch')
    expected_status = 0 if status == 'feasible' else 2 if status == 'infeasible_under_declared_constraints' else None
    if ((expected_status is not None and report['solver_status'] != expected_status)
            or (status == 'inconclusive' and report['solver_status'] not in (1, 3, 4))):
        raise ValueError('Root solver status mismatch')
    if status != 'feasible':
        if 'forces_native' in report:
            raise ValueError('Nonfeasible root result includes a force witness')
        return
    force = np.asarray(report['forces_native'], dtype=float)
    if force.shape != (len(mu), 3) or not np.isfinite(force).all():
        raise ValueError('Invalid root force witness')
    residual = float(np.max(np.abs(C@force.ravel()-b)/np.maximum(np.abs(b), 1e-6)))
    violation = float(max(0., np.max(-force[:, 2]),
                          np.max(np.abs(force[:, 0])+np.abs(force[:, 1])-mu*force[:, 2])))
    if residual > 1e-6 or violation > 1e-6:
        raise ValueError('Root force witness fails original constraints')
    for key, actual in [('maximum_scaled_residual', residual), ('maximum_constraint_violation', violation)]:
        value = report.get(key)
        if (type(value) not in (int, float) or not np.isfinite(value) or value < 0
                or not np.isclose(value, actual, rtol=1e-9, atol=1e-10)):
            raise ValueError('Root force residual metadata mismatch')


def run_study(workspace):
    import importlib.metadata as metadata
    import mujoco as mj
    from .body import Body
    from .root_ab import derive_xml, audit_models, file_hash, write_json
    from .pose_probe import first_contact_height
    from .support_diagnostics import contact_columns, evaluate_support
    from .force_audit import audit_static_forces
    from .transmission import SiteTendonTransmission
    workspace = Path(workspace); out = workspace/'foot-placement'; out.mkdir(exist_ok=False)
    started = time.monotonic(); deadline = started+PROTOCOL['worker_wall_s']
    result = {'schema': 1, 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
              'status': 'running', 'scientific_outcome': 'not_run', 'candidates': [],
              'biological_validation': False, 'walking_claimed': False, 'standing_claimed': False,
              'CNS_executed': False, 'dynamic_trials_executed': 0}
    arrays = {}
    try:
        source_dir = workspace/'source-study'; source_path = source_dir/'result.json'
        source = json.loads(source_path.read_text()); observation = source_dir/'observations.npz'
        if source['status'] != 'completed' or file_hash(observation) != source['observations']['sha256']:
            raise ValueError('Verified complete source study required')
        legacy = Body('muscle_compliance', 'tendon_candidate')
        if legacy.digest != source['body_sha256']:
            raise ValueError('Source model identity mismatch')
        xml = derive_xml(legacy.xml); model = mj.MjModel.from_xml_string(xml)
        invariants = audit_models(legacy.m, model)
        body = Body.__new__(Body); body.m = model; body.d = mj.MjData(model)
        data = body.d; transmission = SiteTendonTransmission(model)
        with np.load(observation, allow_pickle=False) as z:
            q0 = z['neutral_qpos'].copy()
        if q0.shape != (model.nq,) or not np.isfinite(q0).all():
            raise ValueError('Finite matching neutral coordinates required')
        data.qpos[:] = q0; mj.mj_forward(model, data)
        initial_state = body.state(); l0 = data.ten_length.copy()
        fmax = np.asarray(source['maximum_tensions_native'], dtype=float)
        if (model.ntendon != 90 or model.nu != 0 or fmax.shape != (90,)
                or not np.isfinite(fmax).all() or np.any(fmax <= 0) or np.any(l0 <= 0)):
            raise ValueError('Original unactuated 90-tendon model and positive bounds required')
        bodies = []
        for leg in LEGS:
            matches = [i for i in range(1, model.nbody) if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or '').endswith('/'+leg+'_tarsus5')]
            if len(matches) != 1:
                raise ValueError('One terminal tarsus5 body required for '+leg)
            bodies.append(matches[0])
        neutral_feet = data.xpos[bodies].copy()
        receipt = {'parent_body_sha256': legacy.digest, 'candidate_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
                   'source_registration_sha256': file_hash(workspace/'runs/six-leg-tendon-candidate/results.json'),
                   'source_audit_sha256': file_hash(workspace/'runs/source-tendon-audit/results.json'),
                   'asset_sha256': sorted(legacy.asset_files.values()), 'compiled_invariants': invariants,
                   'operation': 'root stiffness/damping removed; no further model change'}
        receipt['model_identity'] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        write_json(out/'model-receipt.json', receipt)
        result.update(model_identity=receipt['model_identity'], model_receipt_sha256=file_hash(out/'model-receipt.json'),
                      source_result_sha256=file_hash(source_path), source_observations_sha256=file_hash(observation),
                      versions={n: metadata.version(n) for n in ('numpy', 'scipy', 'mujoco', 'flygym')}, units=source['units'])
        arrays.update(neutral_qpos=q0, neutral_foot_origins=neutral_feet, neutral_lengths=l0, maximum_tensions=fmax,
                      active_qpos=legacy.active_qpos.copy(), joint_ranges=model.jnt_range[legacy.active_joint_ids].copy())
        candidates = [(q0.copy(), {'index': 0, 'kind': 'unchanged_neutral_control'})]
        # Fixed nine targets, each independently initialized from the same neutral pose.
        for scale in PROTOCOL['xy_scales']:
            for height in PROTOCOL['target_z_offsets_native']:
                targets = neutral_feet.copy()
                targets[:, :2] = q0[:2]+scale*(targets[:, :2]-q0[:2])
                targets[:, 2] = float(neutral_feet[:, 2].min())+height
                q, fit = fit_foot_origins(model, q0, legacy.active_joint_ids, bodies, targets, deadline=deadline)
                row = {'index': len(candidates), 'kind': 'individual_leg_ik', 'xy_scale': scale,
                       'target_z_offset_native': height, 'fit': fit}
                arrays[f'pose_{row["index"]:02d}_targets'] = targets
                candidates.append((q, row))
        for q, row in candidates:
            _deadline(deadline)
            index = row['index']; prefix = f'pose_{index:02d}_'
            arrays[prefix+'candidate_qpos'] = q.copy()
            row['samples'] = []; result['candidates'].append(row)
            try:
                height = first_contact_height(model, data, q, deadline=deadline)
            except ValueError as exc:
                row.update(status='not_bracketed', reason=str(exc)); continue
            row['first_contact_height_native'] = height
            for n, depth in enumerate(PROTOCOL['depths_native']):
                _deadline(deadline)
                data.qpos[:] = q; data.qpos[2] = height-depth
                data.qvel[:] = 0; data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
                mj.mj_forward(model, data)
                if np.any(data.warning.number):
                    raise ValueError('Solver warning in foot-placement probe')
                C, mu, contacts, nonfoot = contact_columns(model, data)
                legs = [leg for leg in LEGS if any('/'+leg+'_tarsus' in c['geom'] for c in contacts)]
                sample = {'index': n, 'depth_native': depth, 'foot_legs': legs, 'nonfoot_contacts': nonfoot,
                          'contacts': contacts, 'root_balance': {'feasible': False, 'status': 'not_eligible'}}
                sp = prefix+f'depth_{n:02d}_'
                arrays.update({sp+'qpos': data.qpos.copy(), sp+'contact_map': C[:6].copy(),
                               sp+'target': (data.qfrc_bias-data.qfrc_passive)[:6].copy(), sp+'friction': mu.copy()})
                if contacts and not nonfoot:
                    sample['root_balance'] = root_balance(arrays[sp+'contact_map'], arrays[sp+'target'], mu)
                row['samples'].append(sample)
            selected = select_sample(row['samples'])
            if selected is None:
                row['status'] = 'no_foot_only_pose'; continue
            row['selected_sample_index'] = selected['index']
            if not selected['root_balance']['feasible']:
                row['status'] = 'root_balance_unresolved'; continue
            sp = prefix+f'depth_{selected["index"]:02d}_'
            data.qpos[:] = arrays[sp+'qpos']; data.qvel[:] = 0
            mj.mj_forward(model, data)
            arrays[prefix+'tendon_lengths'] = data.ten_length.copy()
            limits = fmax*np.exp(-((data.ten_length/l0-1)/.5)**2)
            support, inputs = evaluate_support(body, transmission, limits)
            audit, forces = audit_static_forces(model, data.qpos)
            if not audit['passed'] or not np.allclose(forces['support_target'], inputs['support_target'], rtol=1e-12, atol=1e-12):
                raise ValueError('Selected support force accounting mismatch')
            arrays.update({prefix+k: v for k, v in inputs.items()})
            arrays.update({prefix+'force_'+k: v for k, v in forces.items()})
            row.update(status=support['status'], support=support, force_accounting=audit)
        body.restore(initial_state)
        result['compiled_invariants_after'] = audit_models(legacy.m, model)
        result['source_files_unchanged'] = (legacy.digest == source['body_sha256']
            and all(file_hash(p) == h for p, h in legacy.model_source_hashes.items())
            and all(file_hash(p) == h for p, h in legacy.asset_files.items()))
        if not result['source_files_unchanged'] or not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Source mutation or nonfinite observations')
        _deadline(deadline)
        result.update(status='completed', scientific_outcome='awaiting_evidence_validation')
    except Exception as exc:
        result.update(status='failed', scientific_outcome=('inconclusive' if isinstance(exc, (TimeoutError, MemoryError))
            else 'blocked' if isinstance(exc, ModuleNotFoundError) else 'invalid_experiment'),
            error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if arrays:
            np.savez_compressed(out/'observations.npz', **arrays)
            result['observations'] = {'file': 'observations.npz', 'sha256': file_hash(out/'observations.npz')}
        result['wall_seconds'] = time.monotonic()-started
        write_json(out/'result.json', result)
    return result


def verify(workspace):
    """Replay saved LP inputs. No MuJoCo import, no trust in summary counts."""
    from .root_ab import file_hash, write_json
    from .static_support import solve_support
    from .root_ab_evidence import check_force, check_support, read_json, require
    workspace = Path(workspace); out = workspace/'foot-placement'
    verdict = {'status': 'failed', 'evidence_valid': False, 'scientific_outcome': 'invalid_experiment',
               'walking_claimed': False, 'standing_claimed': False, 'biological_validation': False}
    try:
        result = read_json(out/'result.json')
        require(all(result.get(k) is False for k in ('walking_claimed', 'standing_claimed',
                'CNS_executed', 'biological_validation'))
                and type(result.get('dynamic_trials_executed')) is int
                and result['dynamic_trials_executed'] == 0,
                'unsupported capability claim')
        frozen = read_json(workspace/'protocol.json')
        require(frozen.get('kind') == PROTOCOL['id'] and frozen.get('parameters') == PROTOCOL
                and frozen.get('protocol_sha256') == protocol_hash(), 'pre-execution protocol mismatch')
        if (result['status'] != 'completed' or result['protocol'] != PROTOCOL
                or result['protocol_sha256'] != protocol_hash() or result['source_files_unchanged'] is not True):
            raise ValueError('Complete unchanged-source protocol result required')
        for path, expected in [(out/'model-receipt.json', result['model_receipt_sha256']),
                               (out/'observations.npz', result['observations']['sha256']),
                               (workspace/'source-study/result.json', result['source_result_sha256']),
                               (workspace/'source-study/observations.npz', result['source_observations_sha256'])]:
            if file_hash(path) != expected:
                raise ValueError('Evidence hash mismatch: '+path.name)
        source = read_json(workspace/'source-study/result.json')
        require(source['observations']['sha256'] == result['source_observations_sha256']
                and result['versions'] == source['versions'] and result['units'] == source['units'],
                'source observations, environment or units mismatch')
        if source['status'] != 'completed':
            raise ValueError('Source experiment incomplete')
        receipt = read_json(out/'model-receipt.json')
        invariant = receipt['compiled_invariants']
        require(invariant.get('passed') is True and invariant == result['compiled_invariants_after']
                and invariant.get('changed_arrays') == ['dof_damping', 'jnt_stiffness']
                and receipt.get('operation') == 'root stiffness/damping removed; no further model change',
                'derived model invariants failed')
        identity = receipt.pop('model_identity')
        if (identity != result['model_identity'] or identity != hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
                or receipt['parent_body_sha256'] != source['body_sha256']):
            raise ValueError('Derived model identity mismatch')
        for folder, field in [('six-leg-tendon-candidate', 'source_registration_sha256'), ('source-tendon-audit', 'source_audit_sha256')]:
            if file_hash(workspace/'runs'/folder/'results.json') != receipt[field]:
                raise ValueError('Original registration/audit missing or changed')
        rows = result['candidates']
        if len(rows) != PROTOCOL['candidate_count'] or [r['index'] for r in rows] != list(range(len(rows))):
            raise ValueError('Complete ordered ten-candidate grid required')
        root_count = full_count = samples_checked = 0; max_feet = 0; inconclusive = False
        with np.load(out/'observations.npz', allow_pickle=False) as z:
            if not all(np.isfinite(z[n]).all() for n in z.files):
                raise ValueError('Nonfinite saved observations')
            np.testing.assert_array_equal(z['maximum_tensions'], source['maximum_tensions_native'])
            with np.load(workspace/'source-study/observations.npz', allow_pickle=False) as original:
                np.testing.assert_array_equal(z['neutral_qpos'], original['neutral_qpos'])
            if np.any(z['neutral_lengths'] <= 0):
                raise ValueError('Invalid neutral tendon lengths')
            for row in rows:
                prefix = f'pose_{row["index"]:02d}_'
                q = z[prefix+'candidate_qpos']
                if q.shape != z['neutral_qpos'].shape:
                    raise ValueError('Candidate coordinate shape mismatch')
                if row['index'] == 0:
                    np.testing.assert_array_equal(q, z['neutral_qpos'])
                else:
                    changed = z['active_qpos']
                    require(changed.ndim == 1 and changed.dtype.kind in 'iu' and len(changed) > 0
                            and len(np.unique(changed)) == len(changed)
                            and np.all((changed >= 7) & (changed < len(q)))
                            and z['joint_ranges'].shape == (len(changed), 2), 'invalid active joint coordinates')
                    fixed = np.ones(len(q), dtype=bool); fixed[changed] = False
                    np.testing.assert_array_equal(q[fixed], z['neutral_qpos'][fixed])
                    if (not row['fit']['root_unchanged_during_ik'] or not row['fit']['within_original_joint_limits']
                            or not row['fit']['artificial_initial_placement'] or row['fit']['walking_claimed']
                            or np.any(q[changed] < z['joint_ranges'][:, 0]) or np.any(q[changed] > z['joint_ranges'][:, 1])):
                        raise ValueError('Candidate joint/root invariant failed')
                    expected_scale = PROTOCOL['xy_scales'][(row['index']-1)//3]
                    expected_height = PROTOCOL['target_z_offsets_native'][(row['index']-1)%3]
                    if row['xy_scale'] != expected_scale or row['target_z_offset_native'] != expected_height:
                        raise ValueError('Candidate target grid changed')
                    targets = z['neutral_foot_origins'].copy()
                    targets[:, :2] = q[:2]+expected_scale*(targets[:, :2]-q[:2])
                    targets[:, 2] = z['neutral_foot_origins'][:, 2].min()+expected_height
                    np.testing.assert_array_equal(z[prefix+'targets'], targets)
                if row['status'] == 'not_bracketed':
                    require(not row.get('samples') and 'support' not in row
                            and 'selected_sample_index' not in row, 'unbracketed candidate claims support')
                    inconclusive = True; continue
                samples = row['samples']
                if ([s['index'] for s in samples] != list(range(5))
                        or [s['depth_native'] for s in samples] != PROTOCOL['depths_native']):
                    raise ValueError('Incomplete fixed depth grid')
                for sample in samples:
                    sp = prefix+f'depth_{sample["index"]:02d}_'
                    samples_checked += 1
                    validate_sample(sample, z[sp+'contact_map'], z[sp+'friction'])
                    expected_q = q.copy(); expected_q[2] = row['first_contact_height_native']-sample['depth_native']
                    np.testing.assert_array_equal(z[sp+'qpos'], expected_q)
                    np.testing.assert_allclose(z[sp+'contact_map'], root_contact_map(expected_q, sample['contacts']),
                                               rtol=1e-12, atol=1e-12)
                    if not sample['nonfoot_contacts'] and sample['foot_legs']:
                        max_feet = max(max_feet, len(sample['foot_legs']))
                        validate_root_result(sample['root_balance'], z[sp+'contact_map'], z[sp+'target'], z[sp+'friction'])
                        recalculated = root_balance(z[sp+'contact_map'], z[sp+'target'], z[sp+'friction'])
                        if (recalculated['feasible'] != sample['root_balance']['feasible']
                                or recalculated['status'] != sample['root_balance']['status']):
                            raise ValueError('Root balance does not reproduce')
                        if recalculated['status'] in ('invalid_solver_solution', 'residual_check_failed'):
                            raise ValueError('Invalid root solver result')
                        inconclusive |= recalculated['status'] == 'inconclusive'
                selected = select_sample(samples)
                expected_index = None if selected is None else selected['index']
                if row.get('selected_sample_index') != expected_index:
                    raise ValueError('Selected depth differs from frozen policy')
                if not selected or not selected['root_balance']['feasible']:
                    require('support' not in row and 'force_accounting' not in row
                            and row['status'] == ('no_foot_only_pose' if selected is None else 'root_balance_unresolved'),
                            'ineligible candidate claims muscle support')
                if selected and selected['root_balance']['feasible']:
                    root_count += 1
                    sp = prefix+f'depth_{selected["index"]:02d}_'
                    np.testing.assert_array_equal(z[prefix+'support_qpos'], z[sp+'qpos'])
                    np.testing.assert_allclose(z[prefix+'support_contact_map'][:6], z[sp+'contact_map'], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(z[prefix+'support_target'][:6], z[sp+'target'], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(z[prefix+'support_friction'], z[sp+'friction'])
                    limits = z['maximum_tensions']*np.exp(-((z[prefix+'tendon_lengths']/z['neutral_lengths']-1)/.5)**2)
                    np.testing.assert_allclose(z[prefix+'support_limits'], limits, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(z[prefix+'support_target'], z[prefix+'force_support_target'], rtol=1e-12, atol=1e-12)
                    check_force(row['force_accounting'], z, prefix+'force_', 'unanchored_root', prefix)
                    check_support(row['support'], z, prefix)
                    require(row['status'] == row['support']['status']
                            and row['support']['foot_contacts'] == selected['contacts']
                            and row['support']['nonfoot_contacts'] == selected['nonfoot_contacts']
                            and row['support']['contact_flags_match'] is True
                            and row['support']['feet'] == {leg: leg in selected['foot_legs'] for leg in LEGS}
                            and row['support'].get('walking_claimed') is False
                            and row['support'].get('biological_validation') is False,
                            'full support metadata mismatch')
                    report = solve_support(*(z[prefix+k] for k in ('support_tendon_map', 'support_contact_map',
                                           'support_target', 'support_limits', 'support_friction')))
                    expected_status = 'feasible_at_tested_pose' if report['feasible'] else report['status']
                    if (report['feasible'] != row['support']['feasible']
                            or expected_status != row['support']['status']
                            or report['solver_status'] != row['support']['solver_status']):
                        raise ValueError('Full support does not reproduce')
                    if report['status'] in ('invalid_solver_solution', 'residual_check_failed'):
                        raise ValueError('Invalid full support result')
                    inconclusive |= report['status'] == 'inconclusive'
                    full_count += int(report['feasible'])
        outcome = 'inconclusive' if inconclusive else ('static_pose_found' if full_count else
                  'root_balance_found_muscle_support_unresolved' if root_count else 'no_root_balance_in_fixed_candidates')
        verdict.update(status='completed', evidence_valid=True, scientific_outcome=outcome,
                       candidates_checked=len(rows), depth_samples_checked=samples_checked,
                       root_feasible_candidates=root_count, full_static_feasible_candidates=full_count,
                       maximum_distinct_feet_without_nonfoot_contact=max_feet,
                       result_sha256=file_hash(out/'result.json'), observations_sha256=result['observations']['sha256'],
                       scope='Saved-input replay with the same SciPy/HiGHS family, not independent physics or biological validation')
    except Exception as exc:
        verdict.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(out/'verification.json', verdict)
    return verdict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.workspace) if args.verify else run_study(args.workspace), indent=2))
