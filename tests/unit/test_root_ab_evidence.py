"""Synthetic schema/force fixtures only; these do not simulate a fly."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from organism_core.root_ab import CONDITIONS, TRIALS, compare, write_json, file_hash
from organism_core.root_ab_evidence import check_support, DEPTHS, OFFSETS


def build_evidence(workspace):
    (workspace/'source-study').mkdir()
    q = np.array([0., 0., 0., 1., 0., 0., 0.]); bias = np.array([0., 0., 10., 0., 0., 0.])
    np.savez(workspace/'source-study/observations.npz', neutral_qpos=q, support_qpos=q, support_target=bias)
    source = {'status': 'completed', 'body_sha256': 'synthetic-body', 'maximum_tensions_native': [1.],
              'versions': {'fixture': 'synthetic'}, 'units': {'fixture': 'not animal data'},
              'observations': {'sha256': file_hash(workspace/'source-study/observations.npz')}}
    write_json(workspace/'source-study/result.json', source)
    for name in ('six-leg-tendon-candidate', 'source-tendon-audit'):
        directory = workspace/'runs'/name; directory.mkdir(parents=True)
        write_json(directory/'results.json', {'fixture': name})
    for condition in CONDITIONS:
        directory = workspace/condition; directory.mkdir()
        stiffness = .4 if condition == CONDITIONS[0] else 0.
        damping = .02 if condition == CONDITIONS[0] else 0.
        force = {'qpos': q, 'qvel': np.zeros(6), 'qfrc_bias': bias, 'qfrc_passive': np.zeros(6),
                 'qfrc_spring': np.zeros(6), 'qfrc_damper': np.zeros(6), 'qfrc_gravcomp': np.zeros(6),
                 'qfrc_fluid': np.zeros(6), 'qfrc_applied': np.zeros(6), 'xfrc_applied': np.zeros((1, 6)),
                 'qfrc_actuator': np.zeros(6), 'qfrc_constraint': np.zeros(6), 'inertial_force': -bias,
                 'gravity_bias_from_com_jacobians': bias, 'support_target': bias,
                 'model_jnt_stiffness': np.array([stiffness]), 'model_dof_damping': np.full(6, damping),
                 'model_dof_armature': np.full(6, 1e-6), 'model_dof_frictionloss': np.zeros(6),
                 'model_qpos_spring': q}
        audit = {'passed': True, 'solver_warning_count': 0, 'model_parameters_unchanged': True,
                 'fresh_zero_input_data': True, 'checks': {'gravity_projection': True,
                     'passive_components': True, 'dynamic_force_balance': True},
                 'root_dofs': list(range(6)), 'root_force_budget_native': {'support_target': bias.tolist()},
                 'root_passive_parameters': {'stiffness': stiffness, 'damping': [damping]*6}}
        arrays = {'neutral_qpos': q, 'recorded_qpos': q, 'neutral_lengths': np.ones(1),
                  'maximum_tensions': np.ones(1), 'initial_integration_state': np.zeros(20),
                  'support_qpos': q, 'support_tendon_map': np.zeros((6, 1)),
                  'support_contact_map': np.zeros((6, 0)), 'support_limits': np.ones(1),
                  'support_friction': np.zeros(0), 'support_target': bias}
        arrays.update({'analytic_'+k: v for k, v in force.items()})
        arrays.update({'recorded_force_'+k: v for k, v in force.items()})
        arrays.update(analytic_velocity_qvel=np.ones(6), analytic_velocity_damper=np.full(6, -damping))
        for name in TRIALS:
            arrays[name+'_qpos'] = np.tile(q, (251, 1))
            arrays[name+'_qvel'] = np.zeros((251, 6))
            arrays[name+'_time_s'] = np.arange(251)*.001
        poses = [{'index': i, 'hip_offset_rad': hip, 'knee_offset_rad': knee,
                  'status': 'no_foot_only_pose_in_sampled_depths',
                  'samples': [{'depth_native': d, 'foot_legs': [], 'nonfoot_contacts': []} for d in DEPTHS]}
                 for i, (hip, knee) in enumerate((h, k) for h in OFFSETS for k in OFFSETS)]
        invariant = {'passed': True, 'changed_arrays': ['dof_damping', 'jnt_stiffness']}
        receipt = {'condition': condition, 'parent_body_sha256': source['body_sha256'],
                   'operation': 'identity' if condition == CONDITIONS[0] else 'root stiffness=0, damping=0',
                   'source_registration_sha256': file_hash(workspace/'runs/six-leg-tendon-candidate/results.json'),
                   'source_audit_sha256': file_hash(workspace/'runs/source-tendon-audit/results.json'),
                   'compiled_model_invariants': invariant, 'asset_hashes': ['synthetic-asset'],
                   'xml_sha256': 'synthetic-xml'}
        receipt['model_identity'] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        write_json(directory/'model-receipt.json', receipt)
        np.savez(directory/'observations.npz', **arrays)
        analytic = copy.deepcopy(audit)
        analytic.update(no_active_contacts=True, translation_spring_law=True, velocity_damper_law=True)
        result = {'condition': condition, 'status': 'completed', 'model_identity': receipt['model_identity'],
                  'model_receipt_sha256': file_hash(directory/'model-receipt.json'),
                  'source_result_sha256': file_hash(workspace/'source-study/result.json'),
                  'source_observations_sha256': source['observations']['sha256'],
                  'versions': source['versions'], 'units': source['units'],
                  'compiled_model_invariants': invariant, 'analytic_control': analytic,
                  'recorded_force_accounting': audit, 'static_support': {'feasible': False, 'status': 'not_eligible'},
                  'scan': {'pose_count': 9, 'hip_offsets_rad': OFFSETS, 'knee_offsets_rad': OFFSETS,
                           'poses': poses, 'feasible_pose_count': 0, 'foot_only_pose_count': 0},
                  'observations': {'sha256': file_hash(directory/'observations.npz')},
                  'biological_validation': False, 'walking_claimed': False, 'standing_claimed': False,
                  'CNS_executed': False}
        write_json(directory/'result.json', result)


@pytest.fixture
def evidence(tmp_path):
    build_evidence(tmp_path)
    return tmp_path


def edit_result(workspace, change):
    p = workspace/CONDITIONS[1]/'result.json'; result = json.loads(p.read_text())
    change(result); write_json(p, result)


def test_consistent_fixture_passes_without_physics_backend(evidence):
    report = compare(evidence)
    assert report['scientific_outcome'] == 'valid_zero_feasible'
    assert report['evidence_validation']['passed'] is True


@pytest.mark.parametrize('change', [
    lambda r: r['scan'].update(feasible_pose_count=1),
    lambda r: r['scan'].update(feasible_pose_count=True),
    lambda r: r['scan'].update(foot_only_pose_count=1),
    lambda r: r['scan'].update(poses=[]),
    lambda r: r['scan'].update(pose_count=8),
    lambda r: r['scan'].update(hip_offsets_rad=[-.6, 0., .6]),
    lambda r: r['scan']['poses'][0].update(index=1),
    lambda r: r['scan']['poses'][0].update(knee_offset_rad=.3),
    lambda r: r['scan']['poses'][0]['samples'].pop(),
    lambda r: r['scan']['poses'][0].update(status='feasible_at_tested_pose'),
    lambda r: r['analytic_control'].update(passed=False),
    lambda r: r['analytic_control']['checks'].update(gravity_projection=False),
    lambda r: r['analytic_control'].update(no_active_contacts=False),
    lambda r: r['recorded_force_accounting'].update(solver_warning_count=1),
    lambda r: r['recorded_force_accounting']['root_passive_parameters'].update(stiffness=.4),
    lambda r: r['static_support'].update(feasible=True),
    lambda r: r.update(model_identity='stale-model'),
    lambda r: r.update(walking_claimed=True),
    lambda r: r.update(standing_claimed=True),
    lambda r: r.update(biological_validation=True),
    lambda r: r.update(CNS_executed=True),
])
def test_inconsistent_summaries_rejected_and_old_success_replaced(evidence, change):
    assert compare(evidence)['status'] == 'completed'
    edit_result(evidence, change)
    with pytest.raises(ValueError):
        compare(evidence)
    failure = json.loads((evidence/'comparison.json').read_text())
    assert failure['status'] == 'failed' and failure['comparison_valid'] is False
    assert failure['scientific_outcome'] == 'invalid_comparison'


@pytest.mark.parametrize('path', ['source-study/result.json', 'source-study/observations.npz',
    'runs/source-tendon-audit/results.json', 'runs/six-leg-tendon-candidate/results.json'])
def test_missing_transitive_source_is_not_valid_negative(evidence, path):
    (evidence/path).unlink()
    with pytest.raises(FileNotFoundError):
        compare(evidence)
    assert json.loads((evidence/'comparison.json').read_text())['comparison_valid'] is False


@pytest.mark.parametrize('key', ['recorded_force_qfrc_passive', 'recorded_force_inertial_force',
    'recorded_force_support_target', 'recorded_force_model_dof_armature', 'analytic_velocity_damper'])
def test_changed_raw_force_rejected_even_when_npz_digest_is_updated(evidence, key):
    p = evidence/CONDITIONS[1]/'observations.npz'
    with np.load(p) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    arrays[key].flat[0] += 1.
    np.savez(p, **arrays)
    edit_result(evidence, lambda r: r.update(observations={'sha256': file_hash(p)}))
    with pytest.raises(ValueError):
        compare(evidence)


def test_receipt_content_cannot_change_with_only_outer_digest_update(evidence):
    p = evidence/CONDITIONS[1]/'model-receipt.json'
    receipt = json.loads(p.read_text()); receipt['parent_body_sha256'] = 'wrong-parent'
    write_json(p, receipt)
    edit_result(evidence, lambda r: r.update(model_receipt_sha256=file_hash(p)))
    with pytest.raises(ValueError, match='model identity'):
        compare(evidence)


def test_valid_and_invalid_feasible_witness_use_original_tolerance():
    arrays = {'support_tendon_map': np.zeros((3, 1)), 'support_contact_map': np.eye(3),
              'support_target': np.array([0., 0., 10.]), 'support_limits': np.ones(1),
              'support_friction': np.ones(1)}
    report = {'feasible': True, 'status': 'feasible_at_tested_pose', 'tensions_native': [0.],
              'contact_forces_native': [[0., 0., 10.]]}
    check_support(report, arrays, '')
    report['contact_forces_native'][0][2] = 1.
    with pytest.raises(ValueError, match='witness'):
        check_support(report, arrays, '')
