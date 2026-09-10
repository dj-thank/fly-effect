"""Static-pose force accounting on the unchanged model, not a standing test.

A new MjData is used; caller coordinates and model parameters are never fitted.
The acceleration after mj_forward is generally NONZERO even at zero velocity.
Bias, passive forces, constraint reaction, and inertia must not be conflated.
"""
from __future__ import annotations

import numpy as np


PARAMETERS = (
    'body_mass', 'body_inertia', 'body_gravcomp', 'body_parentid',
    'jnt_type', 'jnt_bodyid', 'jnt_qposadr', 'jnt_dofadr', 'jnt_stiffness',
    'jnt_limited', 'jnt_range', 'dof_damping', 'dof_frictionloss', 'dof_armature',
    'qpos0', 'qpos_spring', 'tendon_stiffness', 'tendon_damping',
    'tendon_lengthspring', 'tendon_frictionloss',
)


def require_unforced_static_state(data):
    """Fail closed rather than silently omitting an external/actuator force.

Call after mj_forward. This validates the LP's declared inputs, not equilibrium.
"""
    for name in ('qpos', 'qvel', 'qfrc_bias', 'qfrc_passive', 'qfrc_applied',
                 'xfrc_applied', 'qfrc_actuator', 'ctrl', 'act'):
        if not np.isfinite(getattr(data, name)).all():
            raise ValueError(f'Static support requires finite {name}')
    if np.any(data.qvel != 0):
        raise ValueError('Static support requires zero velocity')
    for name in ('qfrc_applied', 'xfrc_applied', 'qfrc_actuator', 'ctrl', 'act'):
        if np.any(getattr(data, name) != 0):
            raise ValueError(f'Static support requires zero {name}; force would be omitted')


def audit_static_forces(model, qpos):
    """Recompute a force budget at zero velocity/input on a rigid free-root model.

Returns JSON metadata and NPZ-ready arrays. Generalized vectors retain MuJoCo's
DOF basis; free-root rotational components are NOT labelled world-frame moments.
Gravity is independently projected using each body's COM Jacobian. Constraints
include ALL solver constraints, not only foot/ground contact. Native model units
are preserved; no biological parameter or successful standing claim is inferred.
"""
    import mujoco as mj
    q = np.asarray(qpos, dtype=float).copy()
    if q.shape != (model.nq,) or not np.isfinite(q).all():
        raise ValueError('Finite matching position vector required')
    roots = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if len(roots) != 1 or model.nu or model.nmocap or model.nflex or model.nplugin:
        raise ValueError('Audit requires one rigid free root, no actuators/mocap/flex/plugins')
    root = int(roots[0]); root_body = int(model.jnt_bodyid[root])
    if model.body_parentid[root_body] != 0:
        raise ValueError('Expected free root attached to world')
    # Reject malformed quaternions instead of allowing implicit normalization.
    for joint, kind in enumerate(model.jnt_type):
        if kind in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
            start = int(model.jnt_qposadr[joint]) + (3 if kind == mj.mjtJoint.mjJNT_FREE else 0)
            if not np.isclose(np.linalg.norm(q[start:start+4]), 1., rtol=0., atol=1e-10):
                raise ValueError('Unit joint quaternions required')
    for name in ('passive', 'control'):
        if getattr(mj, 'get_mjcb_' + name)() is not None:
            raise ValueError('Global force/control callbacks are outside audit scope')
    snapshots = {name: np.asarray(getattr(model, name)).copy() for name in PARAMETERS}
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_forward(model, data)
    require_unforced_static_state(data)
    if np.any(data.warning.number):
        raise ValueError('Solver warning during force audit')
    fields = ('qfrc_bias', 'qfrc_passive', 'qfrc_spring', 'qfrc_damper',
              'qfrc_gravcomp', 'qfrc_fluid', 'qfrc_applied', 'xfrc_applied',
              'qfrc_actuator', 'qfrc_constraint', 'qacc', 'qvel', 'qpos')
    arrays = {name: np.asarray(getattr(data, name)).copy() for name in fields}
    # Newer MuJoCo may expose adhesion separately; retain it explicitly if present.
    components = ['qfrc_spring', 'qfrc_damper', 'qfrc_gravcomp', 'qfrc_fluid']
    if hasattr(data, 'qfrc_adhesion'):
        arrays['qfrc_adhesion'] = data.qfrc_adhesion.copy()
        components.append('qfrc_adhesion')
    arrays['passive_component_residual'] = data.qfrc_passive - sum(arrays[n] for n in components)
    gravity = np.asarray(model.opt.gravity).copy()
    disabled = bool(int(model.opt.disableflags) & int(mj.mjtDisableBit.mjDSBL_GRAVITY))
    expected = np.zeros(model.nv)
    if not disabled:
        jac = np.empty((3, model.nv))
        for body in range(1, model.nbody):
            mj.mj_jacBodyCom(model, data, jac, None, body)
            expected -= jac.T @ (model.body_mass[body] * gravity)
    arrays['gravity_bias_from_com_jacobians'] = expected
    arrays['gravity_bias_residual'] = data.qfrc_bias - expected
    arrays['support_target'] = data.qfrc_bias - data.qfrc_passive
    inertia = np.empty(model.nv)
    mj.mj_mulM(model, data, inertia, data.qacc)
    arrays['inertial_force'] = inertia
    arrays['dynamic_balance_residual'] = (inertia + data.qfrc_bias
                                          - data.qfrc_passive - data.qfrc_constraint)
    arrays.update({'model_' + n: v for n, v in snapshots.items()})
    arrays['model_gravity'] = gravity
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite force audit data')
    for name, before in snapshots.items():
        if not np.array_equal(before, getattr(model, name)):
            raise ValueError(f'Force audit mutated model {name}')
    dof = int(model.jnt_dofadr[root]); base = slice(dof, dof+6)
    # Separate force/torque rows: do not normalize mixed physical units together.
    scale = np.maximum.reduce([np.ones(model.nv), np.abs(inertia),
                              np.abs(data.qfrc_bias), np.abs(data.qfrc_passive),
                              np.abs(data.qfrc_constraint)])
    checks = {
        'gravity_projection': np.allclose(data.qfrc_bias, expected, rtol=1e-10, atol=1e-10),
        'passive_components': np.allclose(arrays['passive_component_residual'], 0., atol=1e-10, rtol=0.),
        'dynamic_force_balance': np.all(np.abs(arrays['dynamic_balance_residual'])/scale <= 1e-9),
    }
    root_parameters = {
        'stiffness': float(model.jnt_stiffness[root]),
        'damping': model.dof_damping[base].tolist(),
        'frictionloss': model.dof_frictionloss[base].tolist(),
        'armature': model.dof_armature[base].tolist(),
    }
    report = {
        'kind': 'zero_velocity_force_accounting', 'mujoco_version': mj.__version__,
        'passed': all(checks.values()), 'checks': {k: bool(v) for k, v in checks.items()},
        'model_parameters_unchanged': True, 'fresh_zero_input_data': True,
        'gravity_native': gravity.tolist(), 'gravity_disabled': disabled,
        'total_model_mass_native': float(model.body_mass.sum()),
        'root_subtree_mass_native': float(model.body_subtreemass[root_body]),
        'root_joint': mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, root),
        'root_dofs': list(range(dof, dof+6)), 'root_passive_parameters': root_parameters,
        'root_has_anchor_parameters': bool(root_parameters['stiffness'] != 0
            or np.any(model.dof_damping[base] != 0) or np.any(model.dof_frictionloss[base] != 0)),
        'root_force_budget_native': {n: arrays[n][base].tolist() for n in
            ('qfrc_bias', 'qfrc_passive', *components, 'qfrc_applied',
             'qfrc_actuator', 'qfrc_constraint', 'inertial_force', 'support_target')},
        'maximum_gravity_projection_error': float(np.max(np.abs(arrays['gravity_bias_residual']))),
        'maximum_passive_component_error': float(np.max(np.abs(arrays['passive_component_residual']))),
        'maximum_scaled_dynamic_balance_error': float(np.max(np.abs(arrays['dynamic_balance_residual'])/scale)),
        'maximum_abs_acceleration_native': float(np.max(np.abs(data.qacc))),
        'solver_warning_count': int(data.warning.number.sum()),
        'biological_validation': False, 'walking_claimed': False, 'standing_claimed': False,
        'scope': 'Instantaneous recomputation, not equilibrium or a time average. '
                 'qfrc_constraint includes all constraints. Root components use the MuJoCo DOF basis.',
    }
    return report, arrays
