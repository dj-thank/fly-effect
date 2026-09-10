"""FE-02 root-only A/B: explicit XML derivation, never a runtime default change.

No neural data, checkpoint relabelling, gait controller or force-bound fitting.
The paired treatment removes root stiffness AND damping; armature stays fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time
from xml.etree import ElementTree as ET

import numpy as np

CONDITIONS = ('legacy_anchored', 'unanchored_root_candidate')
TRIALS = ('passive', 'extensor_pulse', 'flexor_pulse', 'tonic_2pct')
PROTECTED = ('organism_core/brain.py', 'organism_core/graph_lock.json', 'ACCEPTANCE.json')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def derive_xml(xml):
    """Only two attributes of one explicit world-attached free joint may change."""
    tree = ET.fromstring(xml)
    roots = tree.findall('./worldbody/body/joint[@type="free"]')
    if len(roots) != 1 or len(tree.findall('.//joint[@type="free"]')) != 1 or tree.findall('.//freejoint'):
        raise ValueError('Exactly one explicit world-attached free joint required')
    root = roots[0]
    if not root.get('name'):
        raise ValueError('Named free root required')
    # Never erase a separate springdamper intervention or silently change defaults.
    for joint in tree.iter('joint'):
        if any(float(v) != 0 for v in joint.get('springdamper', '0 0').split()):
            raise ValueError('Automatic springdamper profile is outside this protocol')
    before = ET.tostring(tree, encoding='unicode')
    old = {key: root.get(key) for key in ('stiffness', 'damping')}
    root.set('stiffness', '0'); root.set('damping', '0')
    candidate = ET.tostring(tree, encoding='unicode')
    for key, value in old.items():
        if value is None:
            root.attrib.pop(key)
        else:
            root.set(key, value)
    # Attribute order is immaterial, but names, geometry and all other data are not.
    if ET.canonicalize(ET.tostring(tree, encoding='unicode')) != ET.canonicalize(before):
        raise ValueError('Unexplained XML change')
    return candidate


def audit_models(legacy, candidate):
    """Compare all public compiled model arrays and solver options, fail closed."""
    import mujoco as mj
    roots = np.flatnonzero(legacy.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if len(roots) != 1:
        raise ValueError('One compiled free root required')
    root = int(roots[0]); dof = int(legacy.jnt_dofadr[root]); base = slice(dof, dof+6)
    if (root != 0 or dof != 0 or legacy.jnt_qposadr[root] != 0
            or legacy.body_parentid[legacy.jnt_bodyid[root]] != 0):
        raise ValueError('Protocol requires world-attached root at qpos[0:7]')
    if not np.isclose(legacy.jnt_stiffness[root], .4, rtol=0, atol=1e-15):
        raise ValueError('Legacy root stiffness differs from preregistration')
    if not np.allclose(legacy.dof_damping[base], .02, rtol=0, atol=1e-15):
        raise ValueError('Legacy root damping differs from preregistration')
    if not np.allclose(legacy.dof_armature[base], 1e-6, rtol=0, atol=1e-18) or np.any(legacy.dof_frictionloss[base]):
        raise ValueError('Legacy root armature/friction differs from preregistration')
    checked = []; changed = []
    for name in dir(legacy):
        if name.startswith('_') or name == 'signature':
            # Compilation identity is not a mechanical parameter (MuJoCo mjModel).
            continue
        left = getattr(legacy, name)
        if isinstance(left, np.ndarray):
            right = getattr(candidate, name)
            expected = left.copy()
            if name == 'jnt_stiffness':
                expected[root] = 0
            elif name == 'dof_damping':
                expected[base] = 0
            if expected.dtype.kind in 'fc' and not np.isfinite(expected).all():
                raise ValueError('Nonfinite model array: '+name)
            if expected.shape != right.shape or not np.array_equal(expected, right):
                raise ValueError('Unexpected compiled model change: '+name)
            checked.append(name)
            if not np.array_equal(left, right):
                changed.append(name)
        elif isinstance(left, (int, float, str, bytes)) and left != getattr(candidate, name):
            raise ValueError('Unexpected model scalar change: '+name)
    for name in dir(legacy.opt):
        if name.startswith('_'):
            continue
        left = getattr(legacy.opt, name)
        if isinstance(left, (np.ndarray, int, float)) and not np.array_equal(left, getattr(candidate.opt, name)):
            raise ValueError('Unexpected solver option change: '+name)
    if changed != ['dof_damping', 'jnt_stiffness']:
        raise ValueError('Declared root intervention was not applied')
    return {'passed': True, 'compiled_arrays_checked': checked, 'changed_arrays': changed,
            'root_joint_id': root, 'root_dofs': list(range(dof, dof+6)),
            'unchanged_armature': candidate.dof_armature[base].tolist(),
            'unchanged_frictionloss': candidate.dof_frictionloss[base].tolist()}


def analytic_control(model, q0):
    """Instantaneous airborne position/velocity probes, not forced locomotion."""
    import mujoco as mj
    from .contacts import is_active_contact
    from .force_audit import audit_static_forces
    q = np.asarray(q0, dtype=float).copy(); q[:3] += [.3, -.2, 10.4]
    report, arrays = audit_static_forces(model, q)
    if not report['passed']:
        raise ValueError('Airborne force accounting failed')
    data = mj.MjData(model); data.qpos[:] = q
    data.qvel[:6] = [.1, -.2, .3, -.4, .5, -.6]
    mj.mj_forward(model, data)
    if any(is_active_contact(c) for c in data.contact[:data.ncon]):
        raise ValueError('Analytic control is not contact-free')
    if np.any(data.warning.number):
        raise ValueError('Analytic control solver warning')
    spring = -model.jnt_stiffness[0]*(q[:3]-model.qpos_spring[:3])
    damper = -model.dof_damping[:6]*data.qvel[:6]
    if not np.allclose(data.qfrc_spring[:3], spring, rtol=1e-12, atol=1e-12):
        raise ValueError('Root translational spring law failed')
    if not np.allclose(data.qfrc_damper[:6], damper, rtol=1e-12, atol=1e-12):
        raise ValueError('Root damper law failed')
    arrays.update(velocity_qpos=data.qpos.copy(), velocity_qvel=data.qvel.copy(),
                  velocity_spring=data.qfrc_spring.copy(), velocity_damper=data.qfrc_damper.copy(),
                  expected_translation_spring=spring, expected_root_damper=damper)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite analytic control')
    report.update(no_active_contacts=True, translation_spring_law=True, velocity_damper_law=True)
    return report, arrays


def run_condition(workspace, condition):
    import mujoco as mj
    from .body import Body
    from .force_audit import audit_static_forces
    from .pose_probe import scan_initial_poses, ground_contacts
    from .support_diagnostics import evaluate_support
    from .tendon_study import _trial
    from .transmission import SiteTendonTransmission
    if condition not in CONDITIONS:
        raise ValueError('Unknown root A/B condition')
    workspace = Path(workspace); out = workspace/condition
    out.mkdir(exist_ok=False)
    started = time.monotonic(); deadline = started+175.
    result = {'schema': 1, 'kind': 'FE-02-root-compliance-ab-v1', 'condition': condition,
              'status': 'running', 'scientific_outcome': 'not_run', 'biological_validation': False,
              'walking_claimed': False, 'standing_claimed': False, 'CNS_executed': False}
    arrays = {}
    try:
        source_dir = workspace/'source-study'; source = json.loads((source_dir/'result.json').read_text())
        observation = source_dir/'observations.npz'
        if source['status'] != 'completed' or file_hash(observation) != source['observations']['sha256']:
            raise ValueError('Completed verified source observations required')
        legacy = Body('muscle_compliance', 'tendon_candidate')
        if legacy.digest != source['body_sha256']:
            raise ValueError('Source model receipt mismatch')
        candidate_xml = derive_xml(legacy.xml)
        candidate = mj.MjModel.from_xml_string(candidate_xml)
        invariants = audit_models(legacy.m, candidate)
        if legacy.m.ntendon != 90 or legacy.m.nu != 0 or abs(legacy.m.opt.timestep-.0001) > 1e-12:
            raise ValueError('Expected unactuated 90-tendon, 100-us model')
        # A deliberately restricted mechanical view; no muscles/brain are constructed.
        body = Body.__new__(Body)
        body.m = legacy.m if condition == CONDITIONS[0] else candidate
        body.d = mj.MjData(body.m); body.joint_geometry = legacy.joint_geometry
        m, d = body.m, body.d
        xml = legacy.xml if condition == CONDITIONS[0] else candidate_xml
        (out/'body.xml').write_text(xml, encoding='utf-8', newline='\n')
        receipt = {'kind': 'root_only_tendon_derivation', 'condition': condition,
                   'parent_body_sha256': legacy.digest, 'xml_sha256': file_hash(out/'body.xml'),
                   'source_registration_sha256': file_hash(workspace/'runs/six-leg-tendon-candidate/results.json'),
                   'source_audit_sha256': file_hash(workspace/'runs/source-tendon-audit/results.json'),
                   'asset_hashes': sorted(legacy.asset_files.values()),
                   'operation': 'identity' if condition == CONDITIONS[0] else 'root stiffness=0, damping=0',
                   'registration': 'Preserved source registration; all compiled tendon/site arrays verified unchanged.',
                   'compiled_model_invariants': invariants, 'biological_validation': False}
        receipt['model_identity'] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        write_json(out/'model-receipt.json', receipt)
        with np.load(observation, allow_pickle=False) as archive:
            q0 = archive['neutral_qpos'].copy(); recorded = archive['support_qpos'].copy()
            source_target = archive['support_target'].copy()
        if any(q.shape != (m.nq,) or not np.isfinite(q).all() for q in (q0, recorded)):
            raise ValueError('Invalid source coordinates')
        d.qpos[:] = q0; mj.mj_forward(m, d); l0 = d.ten_length.copy()
        fmax = np.asarray(source['maximum_tensions_native'], dtype=float)
        if fmax.shape != (90,) or not np.isfinite(fmax).all() or np.any(fmax <= 0):
            raise ValueError('Invalid source tension bounds')
        arrays.update(neutral_qpos=q0, recorded_qpos=recorded, neutral_lengths=l0, maximum_tensions=fmax)
        result.update(model_identity=receipt['model_identity'], model_receipt_sha256=file_hash(out/'model-receipt.json'),
                      source_result_sha256=file_hash(source_dir/'result.json'), source_observations_sha256=file_hash(observation),
                      versions={n: metadata.version(n) for n in ('numpy', 'mujoco', 'flygym', 'scipy')},
                      units=source['units'], compiled_model_invariants=invariants)
        result['analytic_control'], control_arrays = analytic_control(m, q0)
        arrays.update({'analytic_'+k: v for k, v in control_arrays.items()})
        d.qpos[:] = recorded; d.qvel[:] = 0; mj.mj_forward(m, d)
        d.qacc_warmstart[:] = 0  # Equal numerical history as well as equal physical coordinates.
        initial = body.state(); arrays['initial_integration_state'] = initial.copy()
        trans = SiteTendonTransmission(m)
        limits = fmax*np.exp(-((d.ten_length/l0-1)/.5)**2)
        result['static_support'], inputs = evaluate_support(body, trans, limits)
        arrays.update(inputs)
        result['recorded_force_accounting'], forces = audit_static_forces(m, recorded)
        arrays.update({'recorded_force_'+k: v for k, v in forces.items()})
        if not result['recorded_force_accounting']['passed']:
            raise ValueError('Recorded force accounting failed')
        if not np.allclose(forces['support_target'], inputs['support_target'], rtol=1e-12, atol=1e-12):
            raise ValueError('Support target differs from force accounting')
        if condition == CONDITIONS[0] and not np.allclose(inputs['support_target'], source_target, rtol=1e-12, atol=1e-12):
            raise ValueError('Legacy recorded target does not reproduce')
        result['recorded_loads'] = ground_contacts(m, d)
        result['scan'], poses = scan_initial_poses(body, trans, q0, fmax, l0,
                                                  deadline=min(deadline, time.monotonic()+120.))
        arrays.update(poses)
        result['trials'] = []
        for name in TRIALS:
            trial, observations = _trial(body, trans, initial, fmax, l0, .25, name, None, deadline)
            result['trials'].append(trial)
            arrays.update({name+'_'+k: v for k, v in observations.items()})
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite paired observations')
        if time.monotonic() > deadline:
            raise TimeoutError('Condition wall budget exceeded')
        result.update(status='completed', scientific_outcome='awaiting_pair_validation')
    except Exception as exc:
        outcome = 'inconclusive' if isinstance(exc, (TimeoutError, MemoryError)) else (
            'blocked' if isinstance(exc, ModuleNotFoundError) else 'invalid_comparison')
        result.update(status='failed', scientific_outcome=outcome, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if arrays:
            np.savez_compressed(out/'observations.npz', **arrays)
            result['observations'] = {'file': 'observations.npz', 'sha256': file_hash(out/'observations.npz')}
        result['wall_seconds'] = time.monotonic()-started
        write_json(out/'result.json', result)
    return result


def compare_arrays(left, right):
    """Do not confuse a shared list of condition names with matched controls."""
    required = {'neutral_qpos', 'recorded_qpos', 'neutral_lengths', 'maximum_tensions',
                'initial_integration_state', 'support_qpos', 'support_tendon_map',
                'support_contact_map', 'support_limits', 'support_friction', 'support_target'}
    required.update(c+'_'+f for c in TRIALS for f in ('qpos', 'qvel', 'time_s'))
    if not required.issubset(left) or not required.issubset(right):
        raise ValueError('Missing required paired observations')
    if set(left) != set(right):
        raise ValueError('Different observation fields or pose eligibility')
    checked = []
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape:
            raise ValueError('Unmatched observation shape: '+key)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite observation: '+key)
        invariant = key in ('neutral_qpos', 'recorded_qpos', 'neutral_lengths', 'maximum_tensions', 'initial_integration_state') or key.endswith(
            ('support_qpos', 'support_tendon_map', 'support_contact_map', 'support_limits', 'support_friction'))
        if invariant:
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError('Unmatched paired input: '+key)
            checked.append(key)
    for condition in TRIALS:
        for field in ('qpos', 'qvel'):
            key = condition+'_'+field
            if left[key].shape != right[key].shape or not np.array_equal(left[key][0], right[key][0]):
                raise ValueError('Unmatched dynamic initial state: '+key)
        for record in (left, right):
            if not np.array_equal(record[condition+'_qpos'][0], record['recorded_qpos']) or np.any(record[condition+'_qvel'][0]):
                raise ValueError('Trial did not start from the shared zero-velocity recorded pose')
            t = record[condition+'_time_s']
            if (record[condition+'_qpos'].shape != (251, record['recorded_qpos'].size)
                    or record[condition+'_qvel'].ndim != 2 or record[condition+'_qvel'].shape[0] != 251
                    or t.shape != (251,) or not np.allclose(t, np.arange(251)*.001, rtol=0, atol=1e-9)):
                raise ValueError('Trial did not complete the fixed 250ms window')
        if not np.array_equal(left[condition+'_time_s'], right[condition+'_time_s']):
            raise ValueError('Unmatched observation times')
    return checked


def compare(workspace):
    workspace = Path(workspace); results = []; observations = []
    for condition in CONDITIONS:
        directory = workspace/condition; result = json.loads((directory/'result.json').read_text())
        if result['status'] != 'completed' or result['condition'] != condition:
            raise ValueError('Two completed matching condition results required')
        if file_hash(directory/'model-receipt.json') != result['model_receipt_sha256']:
            raise ValueError('Model receipt hash mismatch')
        if file_hash(directory/'observations.npz') != result['observations']['sha256']:
            raise ValueError('Observation hash mismatch')
        with np.load(directory/'observations.npz', allow_pickle=False) as archive:
            observations.append({k: archive[k].copy() for k in archive.files})
        results.append(result)
    a, b = results
    for field in ('source_result_sha256', 'source_observations_sha256', 'versions', 'units'):
        if a[field] != b[field]:
            raise ValueError('Different paired source/environment: '+field)
    checked = compare_arrays(*observations)
    if a['model_identity'] == b['model_identity']:
        raise ValueError('Candidate must have a distinct model identity')
    invalid = {'invalid_solver_solution', 'residual_check_failed'}
    inconclusive = {'inconclusive', 'not_bracketed'}
    statuses = [r['status'] for result in results for r in result['scan']['poses']]
    statuses += [r['static_support']['status'] for r in results]
    if invalid.intersection(statuses):
        raise ValueError('Invalid support result; no capability conclusion')
    allowed = invalid | inconclusive | {'feasible_at_tested_pose', 'infeasible_under_declared_constraints',
                                        'no_foot_only_pose_in_sampled_depths', 'not_eligible'}
    if set(statuses)-allowed:
        raise ValueError('Unknown support status; no capability conclusion')
    outcome = 'inconclusive' if inconclusive.intersection(statuses) else (
        'valid_feasible_pose' if b['scan']['feasible_pose_count'] or b['static_support']['feasible'] else 'valid_zero_feasible')
    report = {'schema': 1, 'kind': 'FE-02-root-compliance-ab-v1', 'status': 'completed',
              'comparison_valid': True, 'scientific_outcome': outcome, 'matched_input_arrays': checked,
              'model_identities': {c: r['model_identity'] for c, r in zip(CONDITIONS, results)},
              'condition_result_sha256': {c: file_hash(workspace/c/'result.json') for c in CONDITIONS},
              'feasible_pose_counts': {c: r['scan']['feasible_pose_count'] for c, r in zip(CONDITIONS, results)},
              'foot_only_pose_counts': {c: r['scan']['foot_only_pose_count'] for c, r in zip(CONDITIONS, results)},
              'biological_validation': False, 'walking_claimed': False, 'standing_claimed': False,
              'CNS_executed': False, 'scope': 'Fixed grid and combined spring/damping intervention only.',
              'next_action': 'Resolve inconclusive measurement before capability claims.' if outcome == 'inconclusive' else (
                  'Preregister separate dynamic holding test.' if outcome == 'valid_feasible_pose' else
                  'Preregister a front/middle/rear foot-placement or force-direction test; do not tune this grid.')}
    write_json(workspace/'comparison.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS)
    args = parser.parse_args()
    if args.condition:
        run_condition(args.workspace, args.condition)
    else:
        compare(args.workspace)
