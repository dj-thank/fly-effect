"""Bounded, matched-control mechanical probes, independent of the full CNS.

The positive/negative pulses are artificial diagnostic inputs, not a neural
walking controller. Neither this runner nor its results certify F01--F16.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time

import numpy as np


def write_result(out, result):
    """Write a strict JSON receipt to an already-created experiment directory."""
    path = Path(out) / 'result.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def run_mechanics_demo(out):
    from .mechanics import synthetic_audit, DEMO_XML
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = synthetic_audit()
    result['fixture_sha256'] = hashlib.sha256(DEMO_XML.encode()).hexdigest()
    result['versions'] = {name: metadata.version(name) for name in ('numpy', 'mujoco')}
    write_result(out, result)
    if not result['passed']:
        raise RuntimeError('Virtual-work positive/negative control failed')
    return result


def run_body_probe(out, *, duration=.02, joint_profile='muscle_compliance',
                   leg='lf', wall_limit=120.):
    """Compare passive, positive-NMJ and negative-NMJ pulses on the same body.

No graph files are read and no parameter is fitted against acceptance seeds.
All controls start from identical physics and muscle snapshots. This probes
whether an artificial motor event produces force and a physical joint response;
it does NOT test weight support, gait, autonomous walking, or biological match.
The wall budget is checked between physics steps, not a hard OS timeout.
"""
    if isinstance(duration, bool) or not np.isfinite(duration) or not 0 < duration <= .1:
        raise ValueError('Probe duration must be finite, positive, and at most 0.1 s')
    if not np.isfinite(wall_limit) or not 0 < wall_limit <= 600:
        raise ValueError('Wall limit must be finite, positive, and at most 600 s')
    if leg not in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh'):
        raise ValueError('Unknown leg')
    if joint_profile not in ('generic', 'muscle_compliance', 'muscle_transfer'):
        raise ValueError('Unknown joint profile')
    ticks = round(duration / .0001)
    if ticks < 1 or abs(duration - ticks * .0001) > 1e-12:
        raise ValueError('Integer 100-us probe duration required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'schema': 1, 'kind': 'organism_candidate_mechanical_probe',
              'status': 'running', 'biological_validation': False,
              'walking_claimed': False, 'functional_gates_automatically_passed': [],
              'graph_read_or_modified': False, 'artificial_motor_input': True,
              'duration_s_per_control': duration, 'leg': leg,
              'joint_profile': joint_profile, 'muscle_model': 'antagonist',
              'scope': 'Matched single-joint NMJ pulse controls; not support, gait, or full-CNS validation.'}
    try:
        import mujoco as mj
        from .body import Body, UNIT_CONTRACT
        from .muscles import PRIORS
        body = Body(joint_profile, 'antagonist')
        result['body_sha256'] = body.digest
        result['model_source_hashes'] = body.model_source_hashes
        result['joint_profile_parameters'] = body.joint_profile
        result['mesh_hashes'] = [{'file_name': Path(path).name, 'sha256': digest}
                                 for path, digest in sorted(body.asset_files.items())]
        result['units'] = UNIT_CONTRACT
        result['muscle_priors'] = PRIORS
        result['versions'] = {name: metadata.version(name) for name in ('numpy', 'mujoco', 'flygym')}
        source = Path(__file__).parent
        result['code_sha256'] = hashlib.sha256(b''.join(
            p.name.encode() + p.read_bytes() for p in sorted(source.glob('*.py')))).hexdigest()
        if abs(body.m.opt.timestep - .0001) > 1e-12:
            raise ValueError('Body timestep differs from the muscle 100-us clock')
        suffix = f'{leg}_trochanterfemur-{leg}_tibia-pitch'
        matches = [i for i, name in enumerate(body.active_joint_names) if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError('Exactly one active knee pitch joint required')
        joint = matches[0]
        result['joint_name'] = body.active_joint_names[joint]
        result['joint_qpos_index'] = int(body.active_qpos[joint])
        result['joint_dof_index'] = int(body.active_dofs[joint])
        mj.mj_forward(body.m, body.d)
        physical = body.state()
        muscles = body.muscles.state()
        controls = {}
        arrays = {}
        for name, polarity in (('passive', None), ('positive', 0), ('negative', 1)):
            body.restore(physical)
            body.muscles.restore(muscles)
            if polarity is not None:
                events = np.zeros_like(body.muscles.pending)
                events[joint, polarity] = 1.  # Artificial one-event NMJ control.
                body.muscles.observe(events)
            start_qpos = body.d.qpos.copy()
            qpos = [start_qpos]
            velocities = [body.d.qvel.copy()]
            torques = []
            maximum_force = 0.
            for tick in range(ticks):
                if time.monotonic() - started > wall_limit:
                    raise TimeoutError('Mechanical probe exceeded its wall budget')
                body.step()
                expected_time = (tick + 1) * .0001
                if abs(body.d.time - expected_time) > 1e-10:
                    raise ValueError('Physics clock mismatch')
                qpos.append(body.d.qpos.copy())
                velocities.append(body.d.qvel.copy())
                torques.append(float(body.muscles.last_torque[joint]))
                maximum_force = max(maximum_force, float(body.muscles.fiber_force.max(initial=0)))
            arrays[name + '_qpos'] = np.asarray(qpos)
            arrays[name + '_qvel'] = np.asarray(velocities)
            arrays[name + '_joint_torque'] = np.asarray(torques)
            controls[name] = {'artificial_nmj_events': 0 if polarity is None else 1,
                              'ticks': ticks, 'initial_torque_native': torques[0],
                              'maximum_muscle_force_native': maximum_force,
                              'joint_displacement_rad': float(body.d.qpos[body.active_qpos[joint]] - start_qpos[body.active_qpos[joint]]),
                              'solver_warning_count': int(body.d.warning.number.sum())}
        arrays['times_s'] = np.arange(ticks + 1) * .0001
        differences = {name: float(np.max(np.abs(arrays[name + '_qvel'] - arrays['passive_qvel'])))
                       for name in ('positive', 'negative')}
        joint_dof = int(body.active_dofs[joint])
        joint_differences = {name: float(np.max(np.abs(
            arrays[name + '_qvel'][:, joint_dof] - arrays['passive_qvel'][:, joint_dof])))
            for name in ('positive', 'negative')}
        matched_starts = all(np.array_equal(arrays[name + suffix][0], arrays['passive' + suffix][0])
                             for name in ('positive', 'negative') for suffix in ('_qpos', '_qvel'))
        checks = {'identical_initial_physics': matched_starts,
                  'passive_has_no_active_muscle_force': controls['passive']['maximum_muscle_force_native'] == 0,
                  'positive_pulse_has_positive_torque': controls['positive']['initial_torque_native'] > 0,
                  'negative_pulse_has_negative_torque': controls['negative']['initial_torque_native'] < 0,
                  'both_pulses_change_selected_joint_velocity': all(value > 1e-12 for value in joint_differences.values()),
                  'no_solver_warnings': all(value['solver_warning_count'] == 0 for value in controls.values())}
        artifact = out / 'observation.npz'
        np.savez_compressed(artifact, **arrays)
        with artifact.open('rb') as stream:
            observation_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        result.update(status='completed' if all(checks.values()) else 'failed',
                      controls=controls, checks=checks,
                      velocity_difference_from_passive=differences,
                      selected_joint_velocity_difference_rad_s=joint_differences,
                      observation={'file': artifact.name, 'sha256': observation_hash})
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      wall_seconds=time.monotonic() - started)
        write_result(out, result)
        raise
    result['wall_seconds'] = time.monotonic() - started
    write_result(out, result)
    if result['status'] != 'completed':
        raise RuntimeError('Mechanical matched-control checks failed; see result.json')
    return result
