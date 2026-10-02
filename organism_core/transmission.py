"""MuJoCo 3.9 straight-site tendon transmission and independent column audits.

Position kinematics must be refreshed by the caller after any qpos change.
CSR topology belongs to mjModel in 3.9, independent of the constraint Jacobian
setting. Construct a new transmission after any model/topology mutation.
"""
from __future__ import annotations
import numpy as np
import mujoco as mj
from .mechanics import site_tendon_paths, pulling_force, _lengths, _tensions


class SiteTendonTransmission:
    """Apply -J.T @ tension without per-segment Python/Cartesian force calls."""
    def __init__(self, model):
        self.model = model
        self.paths = site_tendon_paths(model)
        if not all(hasattr(model, k) for k in ('ten_J_rownnz', 'ten_J_rowadr', 'ten_J_colind')):
            raise RuntimeError('This transmission requires MuJoCo 3.9 model-owned tendon CSR')
        self.rows = np.repeat(np.arange(model.ntendon), model.ten_J_rownnz)
        self.offsets = np.concatenate([np.arange(a, a+n) for a, n in
                                      zip(model.ten_J_rowadr, model.ten_J_rownnz, strict=True)])
        self.columns = model.ten_J_colind[self.offsets].copy()
        self.starts = np.concatenate([p[:-1] for p in self.paths])
        self.ends = np.concatenate([p[1:] for p in self.paths])
        if np.any(self.columns < 0) or np.any(self.columns >= model.nv):
            raise ValueError('Invalid tendon CSR topology')

    def _values(self, data):
        delta = data.site_xpos[self.ends]-data.site_xpos[self.starts]
        if not np.isfinite(delta).all() or np.any(np.einsum('ij,ij->i',delta,delta)<=1e-24):
            raise ValueError('Nonfinite or degenerate tendon segment')
        values = np.asarray(data.ten_J).ravel()[self.offsets]
        if not np.isfinite(values).all() or not np.isfinite(data.ten_length).all() or np.any(data.ten_length <= 1e-12):
            raise ValueError('Invalid tendon kinematics')
        return values

    def matrix(self, data):
        """Return nv x ntendon generalized force per unit positive tension."""
        result = np.zeros((self.model.nv, self.model.ntendon))
        np.add.at(result, (self.columns, self.rows), -self._values(data))
        return result

    def force(self, data, tensions):
        tension = _tensions(tensions, self.model.ntendon)
        return np.bincount(self.columns, weights=-self._values(data)*tension[self.rows],
                           minlength=self.model.nv)


def compare_columns(actual, reference, *, atol=1e-7, rtol=1e-5):
    actual = np.asarray(actual, dtype=float); reference = np.asarray(reference, dtype=float)
    if actual.ndim != 2 or actual.shape != reference.shape or not actual.size:
        raise ValueError('Matching nonempty force matrices required')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite force matrix')
    if not np.isfinite([atol, rtol]).all() or atol <= 0 or rtol < 0:
        raise ValueError('Invalid comparison tolerance')
    error = np.abs(actual-reference)
    tolerance = atol+rtol*np.maximum(np.abs(actual), np.abs(reference))
    columns = np.all(error <= tolerance, axis=0)
    return {'passed': bool(columns.all()), 'per_tendon_passed': columns.tolist(),
            'per_tendon_maximum_absolute_error': error.max(axis=0).tolist(),
            'maximum_absolute_error': float(error.max()),
            'maximum_tolerance_ratio': float((error/tolerance).max()),
            'atol': atol, 'rtol': rtol}


def audit_transmission(model, data, *, epsilon=1e-6):
    """Audit every tendon, all tangent-space DOFs, without mutating source data.

Three methods: MuJoCo's native J, Cartesian site forces, and two independent
central differences of site-path length. One kinematics pair covers ALL tendons
per DOF, rather than repeating nv perturbations for each individual tendon.
"""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Finite positive finite-difference step required')
    transmission = SiteTendonTransmission(model)
    probe = mj.MjData(model); spec = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(model, spec))
    mj.mj_getState(model, data, state, spec); mj.mj_setState(model, probe, state, spec)
    mj.mj_forward(model, probe)
    q0 = probe.qpos.copy()
    native = transmission.matrix(probe)
    cartesian = np.column_stack([pulling_force(model, probe, [path], [1.])
                                 for path in transmission.paths])
    differences = []
    for step in (epsilon, epsilon/2):
        matrix = np.empty_like(native)
        for dof in range(model.nv):
            velocity = np.zeros(model.nv); velocity[dof] = 1.
            lengths = []
            for sign in (1., -1.):
                probe.qpos[:] = q0
                mj.mj_integratePos(model, probe.qpos, velocity, sign*step)
                mj.mj_fwdPosition(model, probe)
                lengths.append(_lengths(probe, transmission.paths))
            matrix[dof] = -(lengths[0]-lengths[1])/(2*step)
        differences.append(matrix)
    comparisons = {'native_vs_cartesian': compare_columns(native, cartesian),
                   'native_vs_finite_difference': compare_columns(native, differences[1]),
                   'finite_difference_convergence': compare_columns(*differences)}
    observable = np.max(np.abs(cartesian), axis=0) > 1e-7
    negative = compare_columns(-native, cartesian)
    negative_detected = bool(observable.any() and not np.any(
        np.asarray(negative['per_tendon_passed'])[observable]))
    result = {'schema': 1, 'kind': 'per_tendon_numerical_audit',
              'biological_validation': False, 'walking_claimed': False,
              'tendons': int(model.ntendon), 'dofs': int(model.nv),
              'epsilon': epsilon, 'comparisons': comparisons,
              'observable_tendons': int(observable.sum()),
              'reversed_observable_columns_rejected': negative_detected,
              'passed': all(v['passed'] for v in comparisons.values()) and negative_detected}
    return result, {'native': native, 'cartesian': cartesian, 'finite_difference': differences[1]}
