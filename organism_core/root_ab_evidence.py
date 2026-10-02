"""Fail-closed consistency checks for FE-02 receipts, not a gait certificate.

Works on retained JSON/NPZ without MuJoCo, meshes or biological data. Hashes
identify accidentally mixed/stale files; they do not authenticate their author.
The simulator remains responsible for physical execution and LP infeasibility.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

OFFSETS = [-.3, 0., .3]
DEPTHS = [.001, .005, .01, .02, .04]
FEASIBLE = 'feasible_at_tested_pose'
NO_POSE = 'no_foot_only_pose_in_sampled_depths'
CHECKS = ('gravity_projection', 'passive_components', 'dynamic_force_balance')
CLAIMS = ('biological_validation', 'walking_claimed', 'standing_claimed', 'CNS_executed')


def require(condition, message):
    if not condition:
        raise ValueError('Evidence validation: '+message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def same(left, right, label):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape and np.array_equal(a, b), 'unmatched '+label)


def close(left, right, label, *, atol=1e-10):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=1e-10, atol=atol), 'inconsistent '+label)


def check_force(report, arrays, prefix, condition, support_prefix=None):
    require(report.get('passed') is True and report.get('solver_warning_count') == 0,
            'force audit failed')
    require(all(report.get('checks', {}).get(k) is True for k in CHECKS), 'force checks failed')
    require(report.get('model_parameters_unchanged') is True
            and report.get('fresh_zero_input_data') is True, 'unverified force-audit state')
    def get(name):
        return arrays[prefix+name]
    bias, passive = get('qfrc_bias'), get('qfrc_passive')
    components = ['qfrc_spring', 'qfrc_damper', 'qfrc_gravcomp', 'qfrc_fluid']
    if prefix+'qfrc_adhesion' in arrays:
        components.append('qfrc_adhesion')
    close(passive, sum(get(k) for k in components), 'passive force sum')
    close(bias, get('gravity_bias_from_com_jacobians'), 'gravity force projection')
    close(get('support_target'), bias-passive, 'support target')
    inertia, reaction = get('inertial_force'), get('qfrc_constraint')
    residual = inertia+bias-passive-reaction
    scale = np.maximum.reduce([np.ones_like(bias), np.abs(inertia), np.abs(bias),
                               np.abs(passive), np.abs(reaction)])
    require(np.isfinite(residual).all() and np.all(np.abs(residual)/scale <= 1e-9),
            'dynamic force balance')
    for name in ('qvel', 'qfrc_applied', 'xfrc_applied', 'qfrc_actuator'):
        require(np.all(get(name) == 0), 'nonzero static input '+name)
    stiffness = .4 if condition == 'legacy_anchored' else 0.
    damping = .02 if condition == 'legacy_anchored' else 0.
    close(get('model_jnt_stiffness')[0:1], [stiffness], 'root stiffness', atol=1e-15)
    close(get('model_dof_damping')[:6], np.full(6, damping), 'root damping', atol=1e-15)
    same(get('model_dof_armature')[:6], np.full(6, 1e-6), 'root armature')
    same(get('model_dof_frictionloss')[:6], np.zeros(6), 'root friction')
    close(get('qfrc_spring')[:3], -stiffness*(get('qpos')[:3]-get('model_qpos_spring')[:3]),
          'root translational spring')
    parameters = report['root_passive_parameters']
    close([parameters['stiffness']], [stiffness], 'reported root stiffness', atol=1e-15)
    close(parameters['damping'], np.full(6, damping), 'reported root damping', atol=1e-15)
    same(report['root_dofs'], list(range(6)), 'root DOFs')
    for name, value in report['root_force_budget_native'].items():
        close(value, get(name)[:6], 'reported root force '+name)
    if support_prefix is not None:
        same(get('qpos'), arrays[support_prefix+'support_qpos'], 'audited support pose')
        close(get('support_target'), arrays[support_prefix+'support_target'], 'audited LP target')


def check_support(report, arrays, prefix):
    status = report['status']
    require(type(report.get('feasible')) is bool and report['feasible'] == (status == FEASIBLE),
            'support feasible flag disagrees with status')
    if status != FEASIBLE:
        return
    # Independently check the claimed feasible witness with the ORIGINAL tolerance.
    T, C, target, limits, mu = [arrays[prefix+'support_'+k] for k in
        ('tendon_map', 'contact_map', 'target', 'limits', 'friction')]
    tension = np.asarray(report['tensions_native'], dtype=float)
    forces = np.asarray(report['contact_forces_native'], dtype=float)
    require(tension.shape == limits.shape and forces.shape == (len(mu), 3), 'support witness shape')
    require(np.isfinite(tension).all() and np.isfinite(forces).all()
            and np.all(limits > 0) and len(mu) > 0, 'invalid support witness')
    scale = np.maximum(np.maximum(np.abs(target), np.max(np.abs(T)*limits, axis=1)), 1e-6)
    residual = T@tension+C@forces.ravel()-target
    require(np.all(np.abs(residual)/scale <= 1e-6)
            and np.all(tension/limits >= -1e-6) and np.all(tension/limits <= 1+1e-6)
            and np.all(forces[:, 2] >= -1e-6)
            and np.all(np.abs(forces[:, 0])+np.abs(forces[:, 1])-mu*forces[:, 2] <= 1e-6),
            'claimed feasible witness fails original constraints')


def check_condition(result, arrays, condition):
    require(all(result.get(k) is False for k in CLAIMS), 'unsupported capability claim')
    check_force(result['analytic_control'], arrays, 'analytic_', condition)
    require(all(result['analytic_control'].get(k) is True for k in
                ('no_active_contacts', 'translation_spring_law', 'velocity_damper_law')),
            'analytic controls incomplete')
    close(arrays['analytic_velocity_damper'][:6],
          -arrays['analytic_model_dof_damping'][:6]*arrays['analytic_velocity_qvel'][:6],
          'velocity damper control')
    check_force(result['recorded_force_accounting'], arrays, 'recorded_force_', condition, '')
    check_support(result['static_support'], arrays, '')
    scan = result['scan']; poses = scan['poses']
    require(type(scan.get('pose_count')) is int and scan['pose_count'] == len(poses) == 9,
            'complete nine-pose grid required')
    require(scan.get('hip_offsets_rad') == OFFSETS and scan.get('knee_offsets_rad') == OFFSETS,
            'preregistered offsets changed')
    for i, ((hip, knee), row) in enumerate(zip(itertools.product(OFFSETS, repeat=2), poses)):
        require(type(row.get('index')) is int and row['index'] == i
                and row.get('hip_offset_rad') == hip and row.get('knee_offset_rad') == knee,
                'pose grid order/identity changed')
        prefix = f'pose_{i:02d}_'
        if row['status'] == 'not_bracketed':
            require('support' not in row and not row.get('samples'), 'unbracketed pose has support')
            require(not any(k.startswith(prefix) for k in arrays), 'unexpected unbracketed arrays')
            continue
        samples = row['samples']
        require([s['depth_native'] for s in samples] == DEPTHS, 'complete five-depth scan required')
        eligible = [s for s in samples if s['foot_legs'] and not s['nonfoot_contacts']]
        if not eligible:
            require(row['status'] == NO_POSE and 'support' not in row, 'ineligible pose claims support')
            require(not any(k.startswith(prefix) for k in arrays), 'unexpected ineligible pose arrays')
            continue
        chosen = min(eligible, key=lambda s: (-len(s['foot_legs']), s['depth_native']))
        require(row.get('selected_depth_native') == chosen['depth_native'], 'selected depth changed')
        require(row['support']['status'] == row['status'], 'pose/support statuses disagree')
        check_support(row['support'], arrays, prefix)
        check_force(row['force_accounting'], arrays, prefix+'force_', condition, prefix)
    for field, expected in (('feasible_pose_count', sum(p['status'] == FEASIBLE for p in poses)),
                            ('foot_only_pose_count', sum('support' in p for p in poses))):
        require(type(scan.get(field)) is int and scan[field] == expected, 'stale summary '+field)
    return {'poses_checked': len(poses), 'force_audits_checked': 2+scan['foot_only_pose_count']}


def validate_evidence(workspace, conditions, results, observations):
    """Require original receipts and recompute counts/force identities before verdict."""
    workspace = Path(workspace)
    source_file = workspace/'source-study/result.json'; source = read_json(source_file)
    source_npz = workspace/'source-study/observations.npz'
    require(source['status'] == 'completed' and digest(source_npz) == source['observations']['sha256'],
            'source observations not verified')
    verified = {}; receipts = []
    with np.load(source_npz, allow_pickle=False) as archive:
        for condition, result, arrays in zip(conditions, results, observations):
            require(result['source_result_sha256'] == digest(source_file)
                    and result['source_observations_sha256'] == digest(source_npz), 'source receipt mismatch')
            require(result['versions'] == source['versions'] and result['units'] == source['units'],
                    'source environment or units changed')
            same(arrays['neutral_qpos'], archive['neutral_qpos'], 'source neutral pose')
            same(arrays['recorded_qpos'], archive['support_qpos'], 'source recorded pose')
            same(arrays['maximum_tensions'], source['maximum_tensions_native'], 'source force bounds')
            if condition == conditions[0]:
                close(arrays['support_target'], archive['support_target'], 'source legacy target')
            directory = workspace/condition; receipt = read_json(directory/'model-receipt.json')
            identity = receipt.pop('model_identity')
            require(identity == result['model_identity']
                    == hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest(),
                    'derived model identity mismatch')
            require(receipt['condition'] == condition and receipt['parent_body_sha256'] == source['body_sha256'],
                    'model parent or condition mismatch')
            expected = 'identity' if condition == conditions[0] else 'root stiffness=0, damping=0'
            require(receipt['operation'] == expected, 'wrong model intervention')
            for key, path in (('source_registration_sha256', 'six-leg-tendon-candidate'),
                              ('source_audit_sha256', 'source-tendon-audit')):
                require(receipt[key] == digest(workspace/'runs'/path/'results.json'), 'source registration mismatch')
            inv = receipt['compiled_model_invariants']
            require(inv == result['compiled_model_invariants'] and inv.get('passed') is True
                    and inv.get('changed_arrays') == ['dof_damping', 'jnt_stiffness'], 'model invariants failed')
            if (directory/'body.xml').exists():
                require(digest(directory/'body.xml') == receipt['xml_sha256'], 'model XML hash mismatch')
            receipts.append(receipt)
            verified[condition] = check_condition(result, arrays, condition)
    same(receipts[0]['asset_hashes'], receipts[1]['asset_hashes'], 'source assets')
    # Root spring/damping may differ; all other saved mechanical parameters must match.
    for key, left in observations[0].items():
        if '_model_' not in key:
            continue
        expected = left.copy()
        if key.endswith('_model_jnt_stiffness'):
            expected[0] = 0
        elif key.endswith('_model_dof_damping'):
            expected[:6] = 0
        same(expected, observations[1][key], 'paired mechanical parameter '+key)
    return {'passed': True, 'conditions': verified,
            'scope': 'Receipt and saved numeric consistency; not physical reexecution or author authentication.'}
