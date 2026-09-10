"""Synthetic consistency controls; these fixtures contain no biological assets."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from organism_core.foot_placement import (
    LEGS, PROTOCOL, protocol_hash, root_balance, root_contact_map,
    validate_root_result, verify,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_fixture():
    q = np.array([0, 0, 0, 1, 0, 0, 0.])
    contacts = [{'position_native': [0, 0, 0]}]
    C = root_contact_map(q, contacts)
    b = np.array([0, 0, 2, 0, 0, 0.]); mu = np.array([.5])
    return root_balance(C, b, mu), C, b, mu


def test_saved_positive_witness_is_valid():
    validate_root_result(*root_fixture())


@pytest.mark.parametrize('mutation', [
    'zero_force', 'wrong_shape', 'nan_force', 'bool_flag', 'status', 'solver',
    'residual', 'negative_residual', 'bool_residual', 'nan_violation', 'violation',
])
def test_bad_saved_witness_or_status_is_rejected(mutation):
    r, C, b, mu = root_fixture()
    if mutation == 'zero_force': r['forces_native'] = [[0., 0., 0.]]
    elif mutation == 'wrong_shape': r['forces_native'] = [0., 0., 2.]
    elif mutation == 'nan_force': r['forces_native'][0][0] = float('nan')
    elif mutation == 'bool_flag': r['feasible'] = 1
    elif mutation == 'status': r['status'] = 'infeasible_under_declared_constraints'
    elif mutation == 'solver': r['solver_status'] = 2
    elif mutation == 'residual': r['maximum_scaled_residual'] = .1
    elif mutation == 'negative_residual': r['maximum_scaled_residual'] = -1
    elif mutation == 'bool_residual': r['maximum_scaled_residual'] = False
    elif mutation == 'nan_violation': r['maximum_constraint_violation'] = float('nan')
    else: r['maximum_constraint_violation'] = .1
    with pytest.raises(ValueError): validate_root_result(r, C, b, mu)


@pytest.mark.parametrize('force', [[1.01, 0, 2], [0, 0, -2], [.6, .6, 2]])
def test_equality_alone_does_not_validate_a_force(force):
    r, C, b, mu = root_fixture()
    r['forces_native'] = [force]
    b = C@force  # Exact equality, but normal or diamond-friction bound is invalid.
    with pytest.raises(ValueError, match='constraints'): validate_root_result(r, C, b, mu)


@pytest.mark.parametrize('mu', [.5, [], [[.5]], [-.5], [float('nan')]])
def test_invalid_friction_input_is_a_validation_error(mu):
    r, C, b, _ = root_fixture()
    with pytest.raises(ValueError): validate_root_result(r, C, b, mu)


@pytest.mark.parametrize('status,solver', [('inconclusive', 1), ('inconclusive', 3),
                                         ('inconclusive', 4), ('infeasible_under_declared_constraints', 2)])
def test_valid_nonfeasible_statuses_stay_distinct(status, solver):
    _, C, b, mu = root_fixture()
    r = {'feasible': False, 'status': status, 'solver_status': solver}
    validate_root_result(r, C, b, mu)
    r['forces_native'] = [[0, 0, 2]]
    with pytest.raises(ValueError): validate_root_result(r, C, b, mu)


@pytest.mark.parametrize('solver', [0, 2, True, 99])
def test_inconclusive_cannot_hide_conflicting_solver_status(solver):
    _, C, b, mu = root_fixture()
    r = {'feasible': False, 'status': 'inconclusive', 'solver_status': solver}
    with pytest.raises(ValueError): validate_root_result(r, C, b, mu)


def test_contact_moments_reconstruct_rotated_translated_root():
    angle = np.pi/3
    q = np.array([4., -3., 2., np.cos(angle/2), 0., 0., np.sin(angle/2)])
    point = np.array([5., -1., 0.]); force = np.array([.2, -.1, .5])
    matrix = root_contact_map(q, [{'position_native': point}])
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    expected = np.r_[force, rotation.T@np.cross(point-q[:3], force)]
    np.testing.assert_allclose(matrix@force, expected, rtol=1e-14, atol=1e-14)
    assert root_contact_map(q, []).shape == (6, 0)


@pytest.mark.parametrize('q', [[], [0]*7, [0, 0, 0, 2, 0, 0, 0],
                               [[0, 0, 0, 1, 0, 0, 0]], [0, 0, np.inf, 1, 0, 0, 0]])
def test_root_map_requires_finite_pose_and_unit_quaternion(q):
    with pytest.raises(ValueError): root_contact_map(q, [])


@pytest.mark.parametrize('point', [[], [1, 2], [0, 0, np.nan]])
def test_root_map_rejects_bad_contact_positions(point):
    with pytest.raises(ValueError): root_contact_map([0, 0, 0, 1, 0, 0, 0], [{'position_native': point}])


def make_workspace(root):
    """Complete 10x5 no-contact fixture; never a model or a positive support test."""
    out = root/'foot-placement'; out.mkdir()
    source = root/'source-study'; source.mkdir()
    q0 = np.array([0., 0., 1., 1., 0., 0., 0., 0.])
    feet = np.zeros((6, 3))
    np.savez(source/'observations.npz', neutral_qpos=q0)
    source_hash = hashlib.sha256((source/'observations.npz').read_bytes()).hexdigest()
    source_record = {'status': 'completed', 'observations': {'sha256': source_hash},
                     'body_sha256': 'synthetic-control', 'maximum_tensions_native': [1.]*90,
                     'versions': {'synthetic': '1'}, 'units': {'length': 'synthetic-unit'}}
    source_result_hash = dump(source/'result.json', source_record)
    invariant = {'passed': True, 'changed_arrays': ['dof_damping', 'jnt_stiffness']}
    receipt = {'parent_body_sha256': 'synthetic-control', 'compiled_invariants': invariant,
               'operation': 'root stiffness/damping removed; no further model change'}
    for directory, field in [('six-leg-tendon-candidate', 'source_registration_sha256'),
                             ('source-tendon-audit', 'source_audit_sha256')]:
        receipt[field] = dump(root/'runs'/directory/'results.json', {'synthetic': True})
    receipt['model_identity'] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    receipt_hash = dump(out/'model-receipt.json', receipt)
    dump(root/'protocol.json', {'kind': PROTOCOL['id'], 'parameters': PROTOCOL, 'protocol_sha256': protocol_hash()})
    arrays = {'neutral_qpos': q0, 'neutral_foot_origins': feet, 'neutral_lengths': np.ones(90),
              'maximum_tensions': np.ones(90), 'active_qpos': np.array([7], dtype=int),
              'joint_ranges': np.array([[-1., 1.]])}
    rows = []
    for i in range(10):
        prefix = f'pose_{i:02d}_'
        row = {'index': i, 'samples': [], 'status': 'no_foot_only_pose', 'first_contact_height_native': 1.}
        arrays[prefix+'candidate_qpos'] = q0.copy()
        if i:
            scale = PROTOCOL['xy_scales'][(i-1)//3]; height = PROTOCOL['target_z_offsets_native'][(i-1)%3]
            row.update(xy_scale=scale, target_z_offset_native=height,
                       fit={'root_unchanged_during_ik': True, 'within_original_joint_limits': True,
                            'artificial_initial_placement': True, 'walking_claimed': False})
            targets = feet.copy(); targets[:, 2] = height; arrays[prefix+'targets'] = targets
        for j, depth in enumerate(PROTOCOL['depths_native']):
            sp = prefix+f'depth_{j:02d}_'; q = q0.copy(); q[2] = 1.-depth
            arrays.update({sp+'qpos': q, sp+'contact_map': np.zeros((6, 0)),
                           sp+'friction': np.zeros(0), sp+'target': np.zeros(6)})
            row['samples'].append({'index': j, 'depth_native': depth, 'contacts': [],
                'foot_legs': [], 'nonfoot_contacts': [], 'root_balance': {'feasible': False, 'status': 'not_eligible'}})
        rows.append(row)
    np.savez(out/'observations.npz', **arrays)
    result = {'status': 'completed', 'protocol': PROTOCOL, 'protocol_sha256': protocol_hash(),
        'source_files_unchanged': True, 'source_result_sha256': source_result_hash,
        'source_observations_sha256': source_hash, 'model_receipt_sha256': receipt_hash,
        'observations': {'sha256': hashlib.sha256((out/'observations.npz').read_bytes()).hexdigest()},
        'versions': source_record['versions'], 'units': source_record['units'],
        'compiled_invariants_after': copy.deepcopy(invariant), 'model_identity': receipt['model_identity'],
        'walking_claimed': False, 'standing_claimed': False, 'biological_validation': False,
        'CNS_executed': False, 'dynamic_trials_executed': 0, 'candidates': rows}
    dump(out/'result.json', result)
    return result


def test_complete_synthetic_negative_grid_is_not_support(tmp_path):
    make_workspace(tmp_path)
    result = verify(tmp_path)
    assert result['evidence_valid'] is True
    assert result['scientific_outcome'] == 'no_root_balance_in_fixed_candidates'
    assert result['depth_samples_checked'] == 50 and result['root_feasible_candidates'] == 0
    assert result['walking_claimed'] is result['standing_claimed'] is False


@pytest.mark.parametrize('mutation', ['walking_claimed', 'standing_claimed', 'biological_validation',
    'CNS_executed', 'dynamic_trials_executed', 'bool_dynamic', 'units', 'versions', 'invariant',
    'unbracketed_support', 'ineligible_support', 'fake_selection', 'source_truthiness', 'protocol'])
def test_verifier_rejects_contradictions_and_replaces_stale_success(tmp_path, mutation):
    result = make_workspace(tmp_path)
    assert verify(tmp_path)['evidence_valid'] is True
    if mutation in ('walking_claimed', 'standing_claimed', 'biological_validation', 'CNS_executed'):
        result[mutation] = True
    elif mutation == 'dynamic_trials_executed': result[mutation] = 1
    elif mutation == 'bool_dynamic': result['dynamic_trials_executed'] = False
    elif mutation == 'units': result['units'] = {'length': 'forged-unit'}
    elif mutation == 'versions': result['versions'] = {'synthetic': '2'}
    elif mutation == 'invariant': result['compiled_invariants_after']['passed'] = False
    elif mutation == 'unbracketed_support':
        result['candidates'][0].update(status='not_bracketed', support={'feasible': True})
    elif mutation == 'ineligible_support': result['candidates'][0]['support'] = {'feasible': True}
    elif mutation == 'fake_selection': result['candidates'][0]['selected_sample_index'] = 0
    elif mutation == 'source_truthiness': result['source_files_unchanged'] = 1
    else: dump(tmp_path/'protocol.json', {'kind': 'other-protocol'})
    dump(tmp_path/'foot-placement/result.json', result)
    with pytest.raises(ValueError): verify(tmp_path)
    verdict = json.loads((tmp_path/'foot-placement/verification.json').read_text())
    assert verdict['evidence_valid'] is False and verdict['status'] == 'failed'
    assert verdict['scientific_outcome'] == 'invalid_experiment'
