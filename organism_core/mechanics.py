"""Independent virtual-work checks for straight, site-routed tendons.

For positive pulling tension F, generalized force is -d(length)/dq * F.
The finite-difference oracle uses MuJoCo's tangent-space position integrator,
so free/ball-joint quaternions are not perturbed as Cartesian coordinates.
This is a numerical mechanics check, not biological validation.
"""
from __future__ import annotations

import numpy as np
import mujoco as mj


def _paths(model, tendons):
    paths = []
    for sites in tendons:
        values = np.asarray(sites)
        if values.ndim != 1 or len(values) < 2 or values.dtype.kind not in 'iu':
            raise ValueError('Each tendon needs at least two integer site IDs')
        if np.any(values < 0) or np.any(values >= model.nsite):
            raise ValueError('Tendon site ID outside model')
        if np.any(values[1:] == values[:-1]):
            raise ValueError('Repeated adjacent tendon site')
        paths.append(values.astype(np.intp, copy=True))
    if not paths:
        raise ValueError('At least one tendon path required')
    return paths


def _tensions(tensions, count):
    values = np.asarray(tensions, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('One finite nonnegative tension per tendon required')
    return values


def _lengths(data, paths):
    result = []
    for sites in paths:
        segments = np.diff(data.site_xpos[sites], axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
            raise ValueError('Nonfinite or degenerate tendon segment')
        result.append(float(lengths.sum()))
    return np.asarray(result)


def pulling_force(model, data, tendons, tensions):
    """Apply site-segment tensions; reject invalid input before applying any force.

The caller must refresh position kinematics after changing qpos. Only straight
site paths are supported, not wrapping surfaces, pulleys, or fixed tendons.
The returned array is independent; qfrc_applied and input arrays are not changed.
"""
    paths = _paths(model, tendons)
    forces = _tensions(tensions, len(paths))
    _lengths(data, paths)  # Validate even zero-tension paths; no silent truncation.
    generalized = np.zeros(model.nv)
    for sites, tension in zip(paths, forces, strict=True):
        if tension == 0:
            continue
        for a, b in zip(sites[:-1], sites[1:], strict=True):
            delta = data.site_xpos[b] - data.site_xpos[a]
            force = float(tension) * delta / np.linalg.norm(delta)
            for site, value in ((a, force), (b, -force)):
                mj.mj_applyFT(model, data, value, np.zeros(3), data.site_xpos[site],
                              int(model.site_bodyid[site]), generalized)
    if not np.isfinite(generalized).all():
        raise ValueError('Nonfinite generalized tendon force')
    return generalized


def site_tendon_paths(model):
    """Read model-ordered paths, refusing geometry our force routine cannot model."""
    paths = []
    for tendon in range(model.ntendon):
        start = int(model.tendon_adr[tendon])
        end = start + int(model.tendon_num[tendon])
        if np.any(model.wrap_type[start:end] != mj.mjtWrap.mjWRAP_SITE):
            raise ValueError('Only straight site-routed tendons are supported')
        paths.append(model.wrap_objid[start:end].copy())
    return _paths(model, paths)


def audit_virtual_work(model, data, tendons, tensions, *, epsilon=1e-6,
                       atol=1e-7, rtol=1e-5, force_function=None):
    """Compare Cartesian-force mapping with an independent length derivative.

The source model and data are not mutated. This function is deliberately an
isolated diagnostic (O(nv) kinematic passes), never part of the 100-us loop.
Failure is reported as passed=False; invalid inputs raise ValueError.
"""
    if not np.isfinite([epsilon, atol, rtol]).all() or epsilon <= 0 or atol <= 0 or rtol < 0:
        raise ValueError('Finite positive epsilon/atol and nonnegative rtol required')
    paths = _paths(model, tendons)
    forces = _tensions(tensions, len(paths))
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError('Finite source state required')
    probe = mj.MjData(model)
    # Copy integration state, then refresh only the private probe's kinematics.
    state_type = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(model, state_type))
    mj.mj_getState(model, data, state, state_type)
    mj.mj_setState(model, probe, state, state_type)
    mj.mj_forward(model, probe)
    neutral = probe.qpos.copy()
    lengths = _lengths(probe, paths)
    implementation = pulling_force if force_function is None else force_function
    actual = np.asarray(implementation(model, probe, paths, forces), dtype=float)
    if actual.shape != (model.nv,) or not np.isfinite(actual).all():
        raise ValueError('Force implementation returned an invalid vector')
    jacobian = np.empty((len(paths), model.nv))
    for dof in range(model.nv):
        velocity = np.zeros(model.nv)
        velocity[dof] = 1.
        samples = []
        for sign in (1., -1.):
            probe.qpos[:] = neutral
            mj.mj_integratePos(model, probe.qpos, velocity, sign * epsilon)
            mj.mj_fwdPosition(model, probe)
            samples.append(_lengths(probe, paths))
        jacobian[:, dof] = (samples[0] - samples[1]) / (2 * epsilon)
    expected = -jacobian.T @ forces
    error = np.abs(actual - expected)
    tolerance = atol + rtol * np.maximum(np.abs(actual), np.abs(expected))
    power = float(actual @ data.qvel)
    expected_power = float(-forces @ (jacobian @ data.qvel))
    return {
        'kind': 'numerical_mechanics', 'biological_validation': False,
        'walking_claimed': False, 'passed': bool(np.all(error <= tolerance)),
        'tendon_count': len(paths), 'degrees_of_freedom': int(model.nv),
        'epsilon': epsilon, 'atol': atol, 'rtol': rtol,
        'lengths_native': lengths.tolist(), 'tensions_native': forces.tolist(),
        'generalized_force_native': actual.tolist(),
        'finite_difference_force_native': expected.tolist(),
        'maximum_absolute_error': float(error.max(initial=0)),
        'maximum_tolerance_ratio': float((error / tolerance).max(initial=0)),
        'mechanical_power_native': power,
        'finite_difference_power_native': expected_power,
        'scope': 'Straight site paths only; virtual work, not force calibration or walking.',
    }


# Original analytical fixture. No fly mesh, connectome, or biological parameter.
DEMO_XML = '''<mujoco model="synthetic_tendon_fixture">
  <option timestep="0.0001" gravity="0 0 0"/>
  <worldbody><body name="base" pos="0 0 1">
    <freejoint/>
    <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>
    <site name="origin" pos="0 1 0"/>
    <body name="limb"><joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" mass="0.1"
            contype="0" conaffinity="0"/>
      <site name="insertion" pos="1 0 0"/>
    </body>
  </body></worldbody>
  <tendon><spatial name="pull"><site site="origin"/><site site="insertion"/></spatial></tendon>
</mujoco>'''


def synthetic_audit():
    """Seed-free, analytical positive control and intentionally reversed negative control."""
    model = mj.MjModel.from_xml_string(DEMO_XML)
    data = mj.MjData(model)
    data.qvel[-1] = .3
    mj.mj_forward(model, data)
    paths = site_tendon_paths(model)
    result = audit_virtual_work(model, data, paths, [2.])
    negative = audit_virtual_work(model, data, paths, [2.],
                                 force_function=lambda *args: -pulling_force(*args))
    result['kind'] = 'synthetic'
    result['analytical_hinge_torque'] = float(np.sqrt(2.))
    result['analytical_match'] = bool(np.isclose(result['generalized_force_native'][-1], np.sqrt(2.)))
    result['internal_base_force_zero'] = bool(np.allclose(result['generalized_force_native'][:6], 0, atol=1e-12))
    result['reversed_force_rejected'] = not negative['passed']
    result['passed'] = all(result[k] for k in ('passed', 'analytical_match',
                                              'internal_base_force_zero', 'reversed_force_rejected'))
    return result
