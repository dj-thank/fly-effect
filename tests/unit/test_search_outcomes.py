"""Scientific outcome regressions using retained records, without physics."""
from copy import deepcopy

import pytest

from organism_core.search_outcomes import (
    FEASIBLE_POSE, INFEASIBLE, full_descent_baseline, full_descent_outcome,
    global_sample_outcome,
)


PROTOCOL = {
    'support_residual_tolerance': 1e-3,
    'n_samples': 2,
    'descend_top_k': 2,
    'certified_poses_per_seed': 2,
    'certified_poses_per_sample': 2,
    'depths_native': [0.001],
}
NEGATIVE = {
    'full': 'descent_floored_without_feasibility',
    'global': 'no_global_improvement',
}
CLASSIFIERS = {'full': full_descent_outcome, 'global': global_sample_outcome}


def _certification():
    return {
        'status': 'completed',
        'accepted_index': 0,
        'any_full_feasible': False,
        'any_passive_feasible': False,
        'samples': [{
            'index': 0,
            'depth_native': 0.001,
            'evidence_saved': True,
            'support': {'status': INFEASIBLE},
            'passive_subsystem': {'status': INFEASIBLE},
            'margin': {'status': 'measured'},
        }],
    }


def _descent():
    return {
        'status': 'step_floor_reached',
        'unresolved_evals': 0,
        'accepted_steps': 0,
        'initial_R': 0.7,
        'accepted': [{'accepted': 0, 'R': 0.7}],
        'final_R': 0.7,
    }


def _result(kind):
    if kind == 'full':
        return {
            'status': 'completed',
            'declared_seeds': 2,
            'seeds': [
                {'status': 'completed', 'descent': _descent(),
                 'certifications': [_certification()]}
                for _ in range(2)
            ],
        }
    result = {
        'status': 'completed',
        'baseline_floor': 0.5,
        'samples': [],
    }
    for index in range(PROTOCOL['n_samples']):
        descent = _descent()
        descent['certifications'] = [_certification()]
        result['samples'].append({
            'index': index, 'status': 'evaluated', 'eligible': True,
            'R': 0.7 + 0.1 * index, 'evidence_saved': True,
            'probe_status': 'infeasible',
            'descent': descent,
        })
    return result


def _records(result, kind):
    return result['seeds' if kind == 'full' else 'samples']


def _certifications(result, kind, index=0):
    record = _records(result, kind)[index]
    return (record['certifications'] if kind == 'full'
            else record['descent']['certifications'])


def _assert_outcome(verdict, expected, *, complete):
    assert verdict['scientific_outcome'] == expected
    assert verdict['outcome_classification']['version'] == 2
    assert verdict['outcome_classification']['search_complete'] is complete
    assert verdict['outcome_classification']['stop_reasons'] == sorted(
        set(verdict['outcome_classification']['stop_reasons']))
    assert verdict['errors'] == []
    if complete:
        assert verdict['outcome_classification']['stop_reasons'] == []
    else:
        assert verdict['outcome_classification']['stop_reasons']


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_completed_step_floor_preserves_scientific_negative(kind):
    verdict = CLASSIFIERS[kind](_result(kind), PROTOCOL)
    _assert_outcome(verdict, NEGATIVE[kind], complete=True)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('status', [
    'wall_budget_exhausted', 'eval_budget_exhausted', 'budget_exhausted',
])
def test_budget_exhaustion_cannot_be_reported_as_completed_floor(kind, status):
    result = _result(kind)
    _records(result, kind)[0]['descent']['status'] = status
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'descent:' + status in verdict['outcome_classification']['stop_reasons']


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_budget_exhausted_certification_prevents_negative(kind):
    result = _result(kind)
    _certifications(result, kind).append({'status': 'budget_exhausted'})
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'certification:budget_exhausted' in verdict['outcome_classification']['stop_reasons']


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_missing_certification_prevents_negative(kind):
    result = _result(kind)
    _certifications(result, kind).clear()
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'certifications_unexecuted' in verdict['outcome_classification']['stop_reasons']


def _partial_certification_work(result, kind, missing):
    """Leave the first descent intact so an independent witness can survive."""
    protocol = deepcopy(PROTOCOL)
    if missing in ('terminal', 'midpoint'):
        record = _records(result, kind)[1]
        record['descent'].update({
            'accepted_steps': 3,
            'initial_R': 0.9,
            'final_R': 0.6,
            'accepted': [
                {'accepted': index, 'R': residual}
                for index, residual in enumerate([0.9, 0.75, 0.65, 0.6])
            ],
        })
        certs = _certifications(result, kind, 1)
        certs[:] = [_certification(), _certification()]
        certs[0]['accepted_index'] = 3
        certs[1]['accepted_index'] = 1
        del certs[0 if missing == 'terminal' else 1]
    else:
        protocol['depths_native'] = [0.001, 0.005]
        for index in range(2):
            cert = _certifications(result, kind, index)[0]
            second_depth = deepcopy(cert['samples'][0])
            second_depth.update(index=1, depth_native=0.005)
            cert['samples'].append(second_depth)
        cert = _certifications(result, kind, 1)[0]
        if missing == 'depth':
            cert['samples'].pop()
        elif missing == 'wrong_depth':
            cert['samples'][1]['depth_native'] = 0.01
        else:
            assert missing == 'duplicate_depth_index'
            cert['samples'][1]['index'] = 0
    return protocol


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('missing', [
    'terminal', 'midpoint', 'depth', 'wrong_depth', 'duplicate_depth_index',
])
def test_incomplete_declared_certification_coverage_prevents_negative(kind, missing):
    result = _result(kind)
    protocol = _partial_certification_work(result, kind, missing)
    verdict = CLASSIFIERS[kind](result, protocol)
    _assert_outcome(verdict, 'inconclusive', complete=False)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('coverage', ['terminal_midpoint', 'depths'])
def test_all_declared_terminal_midpoint_and_depth_work_preserves_negative(kind, coverage):
    result = _result(kind)
    missing = 'terminal' if coverage == 'terminal_midpoint' else 'depth'
    protocol = _partial_certification_work(result, kind, missing)
    certs = _certifications(result, kind, 1)
    if coverage == 'terminal_midpoint':
        terminal = _certification()
        terminal['accepted_index'] = 3
        certs.insert(0, terminal)
    else:
        second_depth = deepcopy(certs[0]['samples'][0])
        second_depth.update(index=1, depth_native=0.005)
        certs[0]['samples'].append(second_depth)
    verdict = CLASSIFIERS[kind](result, protocol)
    _assert_outcome(verdict, NEGATIVE[kind], complete=True)


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_unresolved_descent_evaluation_prevents_negative(kind):
    result = _result(kind)
    _records(result, kind)[0]['descent']['unresolved_evals'] = 1
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'unresolved_descent_evaluations' in verdict['outcome_classification']['stop_reasons']


@pytest.mark.parametrize('declared', [0, None])
def test_no_declared_seeds_does_not_vacuously_prove_a_negative(declared):
    result = {'declared_seeds': declared, 'seeds': []}
    verdict = full_descent_outcome(result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'no_declared_seeds' in verdict['outcome_classification']['stop_reasons']


def test_missing_declared_seed_count_does_not_vacuously_prove_a_negative():
    verdict = full_descent_outcome({'seeds': []}, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'no_declared_seeds' in verdict['outcome_classification']['stop_reasons']


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_unexecuted_declared_work_prevents_negative(kind):
    result = _result(kind)
    _records(result, kind)[1] = {'status': 'not_run'}
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert ('seeds_unexecuted' if kind == 'full' else 'samples_unexecuted') in (
        verdict['outcome_classification']['stop_reasons'])


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_compacted_unexecuted_work_marker_prevents_negative(kind):
    result = _result(kind)
    _records(result, kind)[-1][
        'remaining_seeds_unexecuted' if kind == 'full'
        else 'remaining_samples_unexecuted'] = 3
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)


@pytest.mark.parametrize('descent', [None, {}, {'status': 'budget_exhausted'}])
def test_all_sampling_evaluations_without_topk_descent_are_inconclusive(descent):
    result = _result('global')
    for sample in result['samples']:
        if descent is None:
            del sample['descent']
        else:
            sample['descent'] = deepcopy(descent)
    verdict = global_sample_outcome(result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'samples_unexecuted' not in verdict['outcome_classification']['stop_reasons']
    assert 'certifications_unexecuted' in verdict['outcome_classification']['stop_reasons']


def test_topk_selection_uses_residual_rank_before_checking_completion():
    result = _result('global')
    result['samples'][0]['descent'] = {'status': 'budget_exhausted'}
    protocol = {**PROTOCOL, 'descend_top_k': 1}
    verdict = global_sample_outcome(result, protocol)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'descent:budget_exhausted' in verdict['outcome_classification']['stop_reasons']


def _witness(result, kind, witness):
    cert = _certifications(result, kind)[0]
    if witness == 'full':
        cert['samples'][0]['support']['status'] = FEASIBLE_POSE
        cert['any_full_feasible'] = True
        return 'found_full_support_pose'
    if witness == 'subsystem':
        cert['samples'][0]['passive_subsystem']['status'] = 'feasible'
        cert['any_passive_feasible'] = True
        return 'found_subsystem_feasible_pose'
    if witness == 'tolerance':
        _records(result, kind)[0]['descent']['accepted'][0]['R'] = (
            PROTOCOL['support_residual_tolerance'] / 2)
        return 'tolerance_support_witness'
    assert kind == 'global' and witness == 'basin'
    result['samples'][0]['descent']['final_R'] = 0.25
    return 'global_basin_improvement'


INTERRUPTIONS = [
    'other_descent_budget', 'certification_budget', 'uncertain_saved_evidence',
    'replay_uncertainty', 'unexecuted_work',
]


def _interrupt(result, kind, interruption):
    if interruption == 'other_descent_budget':
        _records(result, kind)[1]['descent']['status'] = 'eval_budget_exhausted'
    elif interruption == 'certification_budget':
        _certifications(result, kind, 1).append({'status': 'budget_exhausted'})
    elif interruption == 'uncertain_saved_evidence':
        _certifications(result, kind, 1)[0]['samples'][0]['margin']['status'] = 'inconclusive'
    elif interruption == 'replay_uncertainty':
        return {'replay_undecided': True}
    else:
        assert interruption == 'unexecuted_work'
        _records(result, kind)[1] = {'status': 'not_run'}
    return {}


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('witness', ['full', 'subsystem', 'tolerance'])
@pytest.mark.parametrize('interruption', INTERRUPTIONS)
def test_completed_positive_survives_other_budget_or_uncertainty(kind, witness, interruption):
    result = _result(kind)
    expected = _witness(result, kind, witness)
    kwargs = _interrupt(result, kind, interruption)
    verdict = CLASSIFIERS[kind](result, PROTOCOL, **kwargs)
    _assert_outcome(verdict, expected, complete=False)


@pytest.mark.parametrize('interruption', INTERRUPTIONS)
def test_global_basin_improvement_survives_other_budget_or_uncertainty(interruption):
    result = _result('global')
    expected = _witness(result, 'global', 'basin')
    kwargs = _interrupt(result, 'global', interruption)
    verdict = global_sample_outcome(result, PROTOCOL, **kwargs)
    _assert_outcome(verdict, expected, complete=False)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('witness', ['full', 'subsystem', 'tolerance'])
@pytest.mark.parametrize('missing', ['terminal', 'midpoint', 'depth'])
def test_completed_witness_survives_other_missing_declared_certification(kind, witness, missing):
    result = _result(kind)
    protocol = _partial_certification_work(result, kind, missing)
    expected = _witness(result, kind, witness)
    verdict = CLASSIFIERS[kind](result, protocol)
    _assert_outcome(verdict, expected, complete=False)


@pytest.mark.parametrize('missing', ['terminal', 'midpoint', 'depth'])
def test_global_basin_improvement_survives_other_missing_declared_certification(missing):
    result = _result('global')
    protocol = _partial_certification_work(result, 'global', missing)
    expected = _witness(result, 'global', 'basin')
    verdict = global_sample_outcome(result, protocol)
    _assert_outcome(verdict, expected, complete=False)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('key,witness', [
    ('any_full_feasible', 'full'), ('any_passive_feasible', 'subsystem'),
])
@pytest.mark.parametrize('actual,stale', [
    (False, True), (True, False), (False, 0), (True, 1),
    (False, None), (False, 'false'),
])
def test_stale_or_non_boolean_certification_summary_is_invalid(kind, key, witness, actual, stale):
    result = _result(kind)
    if actual:
        _witness(result, kind, witness)
    _certifications(result, kind)[0][key] = stale
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    assert verdict['scientific_outcome'] == 'invalid_experiment'
    assert verdict['outcome_classification']['version'] == 2
    assert verdict['outcome_classification']['search_complete'] is False
    assert any(key in error for error in verdict['errors'])


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('key', ['any_full_feasible', 'any_passive_feasible'])
def test_missing_certification_summary_is_invalid(kind, key):
    result = _result(kind)
    del _certifications(result, kind)[0][key]
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    assert verdict['scientific_outcome'] == 'invalid_experiment'
    assert any(key in error for error in verdict['errors'])


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_unsaved_positive_sample_cannot_establish_witness(kind):
    result = _result(kind)
    cert = _certifications(result, kind)[0]
    cert['samples'][0].update({
        'evidence_saved': False,
        'support': {'status': FEASIBLE_POSE},
        'passive_subsystem': {'status': 'feasible'},
        'margin': {'status': 'measured'},
    })
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, NEGATIVE[kind], complete=True)


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_positive_sample_in_unfinished_certification_cannot_establish_witness(kind):
    result = _result(kind)
    _witness(result, kind, 'full')
    _certifications(result, kind)[0]['status'] = 'budget_exhausted'
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)


@pytest.mark.parametrize('kind', CLASSIFIERS)
def test_explicit_replay_false_does_not_reuse_stored_tolerance_witness(kind):
    result = _result(kind)
    _witness(result, kind, 'tolerance')
    assert CLASSIFIERS[kind](result, PROTOCOL)['scientific_outcome'] == 'tolerance_support_witness'
    verdict = CLASSIFIERS[kind](
        result, PROTOCOL, replay_undecided=True, tolerance_witness=False)
    _assert_outcome(verdict, 'inconclusive', complete=False)


def test_explicit_replay_false_does_not_reuse_stored_basin_improvement():
    result = _result('global')
    _witness(result, 'global', 'basin')
    assert global_sample_outcome(result, PROTOCOL)['scientific_outcome'] == 'global_basin_improvement'
    verdict = global_sample_outcome(
        result, PROTOCOL, replay_undecided=True, basin_improved=False)
    _assert_outcome(verdict, 'inconclusive', complete=False)


def test_global_saved_sampling_residual_can_establish_tolerance_witness():
    result = _result('global')
    result['samples'][0]['R'] = PROTOCOL['support_residual_tolerance']
    result['samples'][1]['descent']['status'] = 'budget_exhausted'
    verdict = global_sample_outcome(result, PROTOCOL)
    _assert_outcome(verdict, 'tolerance_support_witness', complete=False)


def test_unsaved_global_sampling_residual_cannot_establish_tolerance_witness():
    result = _result('global')
    result['samples'][0]['R'] = 0.
    result['samples'][0]['evidence_saved'] = False
    verdict = global_sample_outcome(result, PROTOCOL)
    _assert_outcome(verdict, NEGATIVE['global'], complete=True)


@pytest.mark.parametrize('probe_status', ['inconclusive', None])
def test_saved_unresolved_global_probe_prevents_negative_even_if_not_eligible(probe_status):
    result = _result('global')
    result['samples'][0] = {
        'index': 0, 'status': 'evaluated', 'eligible': False, 'R': None,
        'evidence_saved': True,
    }
    if probe_status is not None:
        result['samples'][0]['probe_status'] = probe_status
    verdict = global_sample_outcome(result, PROTOCOL)
    _assert_outcome(verdict, 'inconclusive', complete=False)
    assert 'undecided_saved_probe' in verdict['outcome_classification']['stop_reasons']
    assert 'samples_unexecuted' not in verdict['outcome_classification']['stop_reasons']


@pytest.mark.parametrize('witness', ['full', 'subsystem', 'tolerance', 'basin'])
def test_completed_global_witness_survives_other_saved_unresolved_probe(witness):
    result = _result('global')
    expected = _witness(result, 'global', witness)
    result['samples'][1] = {
        'index': 1, 'status': 'evaluated', 'probe_status': 'inconclusive',
        'eligible': False, 'R': None, 'evidence_saved': True,
    }
    verdict = global_sample_outcome(result, PROTOCOL)
    _assert_outcome(verdict, expected, complete=False)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('root_status', [
    'not_eligible', INFEASIBLE, 'inconclusive', 'residual_check_failed',
    'invalid_solver_solution',
])
def test_unsaved_certification_root_gate_remains_scientific_evidence(kind, root_status):
    result = _result(kind)
    _certifications(result, kind)[0]['samples'][0].update({
        'evidence_saved': False,
        'root_balance': {'feasible': False, 'status': root_status},
    })
    decided = root_status in ('not_eligible', INFEASIBLE)
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(
        verdict, NEGATIVE[kind] if decided else 'inconclusive', complete=decided)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('witness', ['full', 'subsystem', 'tolerance'])
def test_completed_witness_survives_other_unsaved_unresolved_certification_root(kind, witness):
    result = _result(kind)
    expected = _witness(result, kind, witness)
    _certifications(result, kind, 1)[0]['samples'][0].update({
        'evidence_saved': False,
        'root_balance': {'feasible': False, 'status': 'inconclusive'},
    })
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, expected, complete=False)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('residual', [-1., float('inf'), float('nan'), True, '0'])
def test_nonfinite_negative_or_non_numeric_residual_is_not_tolerance_witness(kind, residual):
    result = _result(kind)
    _records(result, kind)[0]['descent']['accepted'][0]['R'] = residual
    verdict = CLASSIFIERS[kind](result, PROTOCOL)
    _assert_outcome(verdict, NEGATIVE[kind], complete=True)


@pytest.mark.parametrize('kind', CLASSIFIERS)
@pytest.mark.parametrize('scenario', ['negative', 'positive', 'inconclusive', 'invalid'])
def test_classifier_leaves_result_and_protocol_unchanged(kind, scenario):
    result = _result(kind)
    if scenario == 'positive':
        _witness(result, kind, 'full')
    elif scenario == 'inconclusive':
        _records(result, kind)[0]['descent']['status'] = 'wall_budget_exhausted'
    elif scenario == 'invalid':
        _certifications(result, kind)[0]['any_full_feasible'] = True
    protocol = deepcopy(PROTOCOL)
    before_result, before_protocol = deepcopy(result), deepcopy(protocol)
    CLASSIFIERS[kind](result, protocol)
    assert result == before_result
    assert protocol == before_protocol


def test_current_verified_inconclusive_search_can_supply_observed_baseline():
    result = _result('full')
    result['seeds'][0]['descent']['status'] = 'eval_budget_exhausted'
    result['seeds'][0]['descent']['final_R'] = 0.6
    result['seeds'][1]['descent']['final_R'] = 0.4
    result['seeds'].append({'status': 'not_run', 'descent': {'final_R': 0.1}})
    verification = full_descent_outcome(result, PROTOCOL)
    verification['evidence_valid'] = True
    assert verification['scientific_outcome'] == 'inconclusive'
    assert verification['outcome_classification']['search_complete'] is False
    before_result, before_verification = deepcopy(result), deepcopy(verification)
    assert full_descent_baseline(result, verification) == 0.4
    assert result == before_result
    assert verification == before_verification


@pytest.mark.parametrize('classification', [None, {'version': 1}])
def test_baseline_requires_current_reverification_even_with_legacy_floor_outcome(classification):
    verification = {
        'evidence_valid': True,
        'scientific_outcome': 'descent_floored_without_feasibility',
    }
    if classification is not None:
        verification['outcome_classification'] = classification
    with pytest.raises(ValueError, match='current outcome re-verification'):
        full_descent_baseline(_result('full'), verification)


@pytest.mark.parametrize('evidence_valid', [False, 1, None])
def test_baseline_requires_verified_evidence_not_truthy_flag(evidence_valid):
    verification = {
        'evidence_valid': evidence_valid,
        'outcome_classification': {'version': 2},
    }
    with pytest.raises(ValueError, match='current outcome re-verification'):
        full_descent_baseline(_result('full'), verification)


def test_baseline_rejects_record_without_finite_nonnegative_completed_residual():
    verification = {
        'evidence_valid': True,
        'outcome_classification': {'version': 2},
    }
    result = {
        'seeds': [
            {'status': 'completed', 'descent': {'final_R': residual}}
            for residual in [float('inf'), float('nan'), -0.1, True, '0', None]
        ] + [{'status': 'not_run', 'descent': {'final_R': 0.2}}],
    }
    with pytest.raises(ValueError, match='no finite observed terminal residual'):
        full_descent_baseline(result, verification)
