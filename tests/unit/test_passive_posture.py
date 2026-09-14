"""Synthetic posture-search fixtures; none represent a validated animal."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from organism_core import passive_posture as pp


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def placement_fixture(root):
    """Reuse the ten-candidate placement workspace; no biological assets."""
    spec = importlib.util.spec_from_file_location('placement_fixture',
        Path(__file__).with_name('test_foot_placement_evidence.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.make_workspace(root)
    return module


def problem():
    # Root row: z1+z2=1; passive rows: z1=1 and z2=1 (jointly inconsistent).
    T = np.zeros((8, 2))
    C = np.zeros((8, 6)); C[2, [2, 5]] = 1; C[6, 2] = 1; C[7, 5] = 1
    b = np.zeros(8); b[[2, 6, 7]] = 1
    return T, C, b, np.ones(2), np.zeros(2)


def test_passive_rows_exact_zero_minus_root():
    T, *_ = problem()
    T[3, 0] = 1e-25  # Roundoff-scale nonzero is NOT a passive row.
    assert pp.passive_rows(T, [0, 1, 2, 3, 4, 5]) == [6, 7]
    with pytest.raises(ValueError): pp.passive_rows(T, [0, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError): pp.passive_rows(T, [0, 1, 2, 3, 4, 8])
    with pytest.raises(ValueError): pp.passive_rows(T, [0., 1, 2, 3, 4, 5])
    bad = T.copy(); bad[6, 0] = np.nan
    with pytest.raises(ValueError): pp.passive_rows(bad, [0, 1, 2, 3, 4, 5])


def trial_inputs():
    q = np.array([0., 0., 1., 1., 0., 0., 0., 0.5, 0.5])
    adrs = np.array([7, 8])
    ranges = np.array([[-1., 1.], [-1., 1.]])
    return q, adrs, ranges


def test_declared_trials_are_deterministic_and_inside_limits():
    q, adrs, ranges = trial_inputs()
    first = pp.trial_postures(q, adrs, ranges, 3)
    second = pp.trial_postures(q, adrs, ranges, 3)
    assert len(first) == len(second) == 1+2*pp.PROTOCOL['dof_grid_points']+pp.PROTOCOL['seeded_box_samples']
    for (a, am), (b, bm) in zip(first, second):
        np.testing.assert_array_equal(a, b); assert am == bm
    assert first[0][1] == {'kind': 'control'}
    np.testing.assert_array_equal(first[0][0], q)
    kinds = [m['kind'] for _, m in first]
    assert kinds.count('control') == 1 and kinds.count('sweep') == 10 and kinds.count('seeded') == 8
    metas = [json.dumps(m, sort_keys=True) for _, m in first]
    assert len(set(metas)) == len(metas)
    for q_trial, _ in first:
        np.testing.assert_array_equal(q_trial[:7], q[:7])
        assert np.all(q_trial[adrs] >= ranges[:, 0]) and np.all(q_trial[adrs] <= ranges[:, 1])
    other = pp.trial_postures(q, adrs, ranges, 4)
    assert not any(np.array_equal(a, b) for (a, ma), (b, mb)
                   in zip([x for x in first if x[1]['kind'] == 'seeded'],
                          [x for x in other if x[1]['kind'] == 'seeded']))


@pytest.mark.parametrize('mutation', ['shape', 'nan_q', 'nan_range', 'empty_range',
                                      'dup_adr', 'root_adr', 'out_of_range_adr', 'outside_limit'])
def test_invalid_search_inputs_rejected(mutation):
    q, adrs, ranges = trial_inputs()
    if mutation == 'shape': ranges = ranges[:, 0]
    elif mutation == 'nan_q': q = q.copy(); q[7] = np.nan
    elif mutation == 'nan_range': ranges = ranges.copy(); ranges[0, 0] = np.nan
    elif mutation == 'empty_range': ranges = ranges.copy(); ranges[0] = [1., -1.]
    elif mutation == 'dup_adr': adrs = np.array([7, 7])
    elif mutation == 'root_adr': adrs = np.array([6, 8])
    elif mutation == 'out_of_range_adr': adrs = np.array([7, 9])
    else: q = q.copy(); q[7] = 2.
    with pytest.raises(ValueError): pp.trial_postures(q, adrs, ranges, 0)


def test_composed_trial_uses_best_recorded_residual():
    q, adrs, ranges = trial_inputs()
    trials = [{'meta': {'kind': 'sweep', 'qpos_adr': 7, 'grid_point': 0, 'value': -.9},
               'samples': [{'passive_subsystem': {'status': 'infeasible_under_declared_constraints',
                    'nearest_balance': {'minimum_scaled_balance_error': .5}}}]},
              {'meta': {'kind': 'sweep', 'qpos_adr': 7, 'grid_point': 1, 'value': -.4},
               'samples': [{'passive_subsystem': {'status': 'infeasible_under_declared_constraints',
                    'nearest_balance': {'minimum_scaled_balance_error': .2}}}]},
              {'meta': {'kind': 'sweep', 'qpos_adr': 8, 'grid_point': 0, 'value': .9},
               'samples': []}]
    q_c, meta = pp.compose_trial(q, trials)
    assert q_c[7] == -.4 and q_c[8] == .5  # Unscored DOF keeps the saved value.
    assert meta == {'kind': 'composed', 'scored_dofs': [7]}
    q2, meta2 = pp.compose_trial(q, trials)
    np.testing.assert_array_equal(q_c, q2); assert meta == meta2
    trials[0]['samples'][0]['passive_subsystem'] = {'status': 'feasible'}
    q3, _ = pp.compose_trial(q, trials)
    assert q3[7] == -.9  # A feasible gate scores zero and wins.


def test_solve_subsystem_marks_scope_and_rows():
    T, C, b, limit, mu = problem()
    report = pp.solve_subsystem(T, C, b, limit, mu, [0, 1, 2, 3, 4, 5])
    assert report['status'] == 'feasible' and report['subsystem_rows'] == list(range(6))
    conflicted = pp.solve_subsystem(T, C, b, limit, mu, list(range(8)))
    assert conflicted['status'] == pp.INFEASIBLE


def screen_args():
    T, C, b, limit, mu = problem()
    return dict(tendon_map=T, contact_map=C, target=b, limits=limit, friction=mu,
                root_rows=[0, 1, 2, 3, 4, 5], passive=[6, 7],
                support_qpos=np.array([0., 0., 0., 1., 0., 0., 0., -.9999, -.9999]),
                qpos_adrs=np.array([7, 8]), stiffness=np.ones(2),
                springref=np.zeros(2), ranges=np.array([[-2., 2.], [-2., 2.]]),
                wall_seconds=30., max_calls=160)


def test_spring_screen_resolves_rows_inside_spring_authority():
    r = pp._spring_screen(**screen_args())
    assert r['status'] == 'completed' and r['physical'] is False
    assert r['rows_that_can_resolve'] == 2
    assert all(e['can_resolve'] and len(e['points']) == 1 for e in r['rows'])


def test_spring_screen_base_cases():
    args = screen_args()
    args['target'] = args['target'].copy(); args['target'][6] = 0
    r = pp._spring_screen(**args)
    assert r['status'] == 'already_feasible' and r['solver_calls'] == 1
    args = screen_args(); args['max_calls'] = 1
    assert pp._spring_screen(**args)['status'] == 'call_budget_exhausted'
    args = screen_args(); args['wall_seconds'] = 0
    with pytest.raises(ValueError): pp._spring_screen(**args)
    args = screen_args()
    args['ranges'] = np.array([[-.001, .001], [-.001, .001]])  # Cannot cover the gap.
    r = pp._spring_screen(**args)
    assert r['status'] == 'completed' and r['rows_that_can_resolve'] == 0


def make_posture_workspace(root):
    """Fabricate a verified-consistent search result over the placement fixture."""
    placement_fixture(root)
    placed_path = root/'foot-placement/result.json'
    placed = json.loads(placed_path.read_text())
    npz_path = root/'foot-placement/observations.npz'
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    # Candidate 0 becomes a two-passive-joint synthetic with a root+passive conflict.
    q = np.array([0., 0., 1., 1., 0., 0., 0., .5, .5])
    T, C, b, limit, mu = problem()
    arrays['pose_00_candidate_qpos'] = q
    arrays['pose_00_support_qpos'] = q.copy()
    arrays['pose_00_support_tendon_map'] = T
    arrays['pose_00_support_contact_map'] = C
    arrays['pose_00_support_target'] = b
    arrays['pose_00_support_limits'] = limit
    arrays['pose_00_support_friction'] = mu
    np.savez(npz_path, **arrays)
    placed['candidates'][0].update(
        status=pp.INFEASIBLE, support={'status': pp.INFEASIBLE},
        force_accounting={'root_dofs': [0, 1, 2, 3, 4, 5]})
    placed_hash = dump(placed_path, placed)
    npz_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    out = root/'passive-posture'; out.mkdir()
    q0 = np.array([0., 0., 1., 1., 0., 0., 0., 0.])
    my = {'neutral_qpos': q0, 'neutral_lengths': np.ones(90), 'maximum_tensions': np.ones(90),
          'pose_00_passive_dofs': np.array([6, 7]), 'pose_00_passive_qpos_adrs': np.array([7, 8]),
          'pose_00_passive_ranges': np.array([[-1., 1.], [-1., 1.]]),
          'pose_00_passive_stiffness': np.array([2., 2.]),
          'pose_00_passive_springref': np.zeros(2)}
    expected = pp.trial_postures(q, np.array([7, 8]), np.array([[-1., 1.], [-1., 1.]]), 0)
    screen = {'kind': 'fixed_geometry_spring_authority_screen', 'physical': False,
              'status': 'completed', 'base_status': pp.INFEASIBLE,
              'solver_calls': 0, 'rows': [], 'rows_that_can_resolve': 0}
    rows = list(range(6))+[6, 7]
    base = pp.solve_subsystem(T, C, b, limit, mu, rows)
    assert base['status'] == pp.INFEASIBLE; screen['solver_calls'] += 1
    for r_i, dof in enumerate([6, 7]):
        lo, hi = -.9999, .9999
        entry = {'dof': dof, 'qpos_adr': 7+r_i, 'stiffness': 2., 'springref': 0.,
                 'range': [-1., 1.], 'current_q': .5, 'can_resolve': False, 'points': []}
        for q_new in np.linspace(lo, hi, pp.PROTOCOL['spring_shift_points']):
            adjusted = b.copy(); adjusted[dof] += 2.*(q_new-.5)
            replay = pp.solve_support(T[rows], C[rows], adjusted[rows], limit, mu)
            screen['solver_calls'] += 1
            entry['points'].append({'q': float(q_new), 'delta': float(2.*(q_new-.5)),
                                    'status': replay['status'],
                                    'minimum_scaled_balance_error':
                                        replay.get('nearest_balance', {}).get(
                                            'minimum_scaled_balance_error')})
            if replay['status'] == 'feasible':
                entry['can_resolve'] = True; break
        screen['rows'].append(entry)
    screen['rows_that_can_resolve'] = sum(e['can_resolve'] for e in screen['rows'])
    trials = []
    for t_i, (q_trial, meta) in enumerate(expected):
        if t_i == 0:
            record = {'meta': meta, 'solver_calls': 0, 'status': 'no_foot_only_pose',
                      'first_contact_height_native': 1., 'selected_sample_index': None,
                      'best_gate': 0, 'best_gate_name': 'no_foot_only_contact',
                      'any_passive_feasible': False, 'any_full_feasible': False,
                      'samples': []}
            tp = 'pose_00_trial_000_'; my[tp+'qpos'] = q_trial
            for j, depth in enumerate(pp.PROTOCOL['depths_native']):
                sp = tp+f'depth_{j:02d}_'; qd = q_trial.copy(); qd[2] = 1.-depth
                my.update({sp+'qpos': qd, sp+'contact_map': np.zeros((6, 0)),
                           sp+'friction': np.zeros(0), sp+'target': np.zeros(6)})
                record['samples'].append({'index': j, 'depth_native': depth, 'contacts': [],
                    'foot_legs': [], 'nonfoot_contacts': [], 'gate_index': 0,
                    'root_balance': {'feasible': False, 'status': 'not_eligible'}})
            trials.append(record)
        else:
            trials.append({'meta': meta, 'status': 'not_bracketed',
                           'reason': 'First ground contact not bracketed within +/-2 native length units'})
    composed_q, composed_meta = pp.compose_trial(q, trials)
    trials.append({'meta': composed_meta, 'status': 'not_bracketed',
                   'reason': 'First ground contact not bracketed within +/-2 native length units'})
    entries = [{'index': 0, 'placement_status': pp.INFEASIBLE, 'status': 'completed',
                'passive_dofs': [6, 7], 'passive_joint_names': ['a', 'b'],
                'spring_screen': screen, 'declared_trials': len(expected)+1,
                'trials': trials, 'solver_calls': 0, 'trials_executed': 1,
                'best_gate': 0, 'passive_feasible_trials': 0, 'full_feasible_trials': 0}]
    for i in range(1, 10):
        entries.append({'index': i, 'placement_status': 'no_foot_only_pose',
                        'status': 'no_root_feasible_placement', 'trials': [], 'solver_calls': 0})
    inputs = ['foot-placement/result.json', 'foot-placement/observations.npz',
              'foot-placement/model-receipt.json', 'protocol.json',
              'source-study/result.json', 'source-study/observations.npz']
    input_hashes = {rel: hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in inputs}
    source = json.loads((root/'source-study/result.json').read_text())
    receipt = json.loads((root/'foot-placement/model-receipt.json').read_text())
    frozen = {'schema': 1, 'protocol': pp.PROTOCOL, 'protocol_sha256': pp.protocol_hash(),
              'input_sha256': input_hashes, 'code_sha256': {'synthetic': 'x'},
              'versions': source['versions']}
    dump(out/'protocol.json', frozen)
    np.savez(out/'observations.npz', **my)
    result = {'schema': 1, 'protocol': pp.PROTOCOL, 'protocol_sha256': pp.protocol_hash(),
              'status': 'completed', 'scientific_outcome': 'no_passive_equilibrium_in_bounded_search',
              'candidates': entries, 'candidates_processed': 10,
              'biological_validation': False, 'walking_claimed': False,
              'standing_claimed': False, 'CNS_executed': False,
              'dynamic_trials_executed': 0, 'new_physics_trajectories': 0,
              'model_identity': receipt['model_identity'],
              'placement_result_sha256': placed_hash,
              'placement_observations_sha256': npz_hash,
              'input_sha256': input_hashes, 'code_sha256': {'synthetic': 'x'},
              'versions': source['versions'], 'units': source['units'],
              'observations': {'file': 'observations.npz',
                               'sha256': hashlib.sha256((out/'observations.npz').read_bytes()).hexdigest()},
              'wall_seconds': .1}
    dump(out/'result.json', result)
    return result


def test_verifier_accepts_consistent_bounded_negative(tmp_path):
    make_posture_workspace(tmp_path)
    verdict = pp.verify(tmp_path)
    assert verdict['evidence_valid'] is True and verdict['status'] == 'completed'
    assert verdict['scientific_outcome'] == 'no_passive_equilibrium_in_bounded_search'
    assert verdict['trials_checked'] == 1 and verdict['witnesses_checked'] == 0
    assert verdict['walking_claimed'] is verdict['standing_claimed'] is False


@pytest.mark.parametrize('mutation', ['walking_claimed', 'standing_claimed',
    'biological_validation', 'CNS_executed', 'dynamic_trials_executed', 'trajectories',
    'protocol', 'outcome', 'limits', 'composed_meta', 'screen_flag', 'input_stale',
    'summary_count', 'ordering'])
def test_verifier_rejects_contradictions(tmp_path, mutation):
    result = make_posture_workspace(tmp_path)
    assert pp.verify(tmp_path)['evidence_valid'] is True
    path = tmp_path/'passive-posture/result.json'
    if mutation in ('walking_claimed', 'standing_claimed', 'biological_validation', 'CNS_executed'):
        result[mutation] = True
    elif mutation == 'dynamic_trials_executed': result['dynamic_trials_executed'] = 1
    elif mutation == 'trajectories': result['new_physics_trajectories'] = 1
    elif mutation == 'protocol': result['protocol'] = dict(result['protocol'], seed=9)
    elif mutation == 'outcome': result['scientific_outcome'] = 'full_support_feasible_posture_found'
    elif mutation == 'summary_count': result['candidates'][0]['trials_executed'] = 20
    elif mutation == 'ordering': result['candidates'][0]['trials'][1]['meta'] = {'kind': 'control'}
    elif mutation == 'composed_meta':
        result['candidates'][0]['trials'][-1]['meta'] = {'kind': 'composed', 'scored_dofs': [7]}
    elif mutation == 'screen_flag': result['candidates'][0]['spring_screen']['rows'][0]['can_resolve'] = False
    elif mutation == 'input_stale':
        placed_path = tmp_path/'foot-placement/result.json'
        placed = json.loads(placed_path.read_text()); placed['touched'] = True
        dump(placed_path, placed)
    elif mutation == 'limits':
        npz = tmp_path/'passive-posture/observations.npz'
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        arrays['pose_00_trial_000_qpos'][7] = 99.
        np.savez(npz, **arrays)
        result['observations']['sha256'] = hashlib.sha256(npz.read_bytes()).hexdigest()
    if mutation != 'input_stale':
        dump(path, result)
    with pytest.raises((ValueError, AssertionError)):
        pp.verify(tmp_path)
    verdict = json.loads((tmp_path/'passive-posture/verification.json').read_text())
    assert verdict['evidence_valid'] is False and verdict['status'] == 'failed'


def test_unverifiable_positive_gate_is_rejected(tmp_path):
    result = make_posture_workspace(tmp_path)
    sample = result['candidates'][0]['trials'][0]['samples'][0]
    sample['passive_subsystem'] = {'status': 'feasible', 'solver_status': 0,
                                   'subsystem_rows': list(range(8))}
    sample['support'] = {'status': 'infeasible_under_declared_constraints', 'solver_status': 2}
    sample['evidence_saved'] = False  # Positive claim without saved matrices must fail.
    dump(tmp_path/'passive-posture/result.json', result)
    with pytest.raises(ValueError): pp.verify(tmp_path)
