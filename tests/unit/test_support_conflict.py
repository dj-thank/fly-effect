"""Synthetic constraint systems; none represent a validated animal."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from organism_core import support_conflict as sc


def problem():
    # Root: z1+z2=1; passive row1: z1=1; passive row2: z2=1.
    # Each passive row is feasible alone; their conjunction is inconsistent.
    T = np.zeros((3, 1))
    C = np.zeros((3, 6)); C[0, [2, 5]] = 1; C[1, 2] = 1; C[2, 5] = 1
    return T, C, np.ones(3), np.ones(1), np.zeros(2)


def call(args=None, **kwargs):
    return sc.diagnose(*(problem() if args is None else args), root_rows=(0,), **kwargs)


def test_joint_conflict_with_no_individually_bad_row_and_rechecked_witnesses():
    args = problem(); saved = [x.copy() for x in args]
    r = call(args)
    assert r['status'] == 'passive_conflict_isolated'
    assert r['conflict_rows'] == [1, 2] and r['inclusion_minimal'] is True
    assert all(x['status'] == 'feasible' for x in r['single_rows'])
    assert all(x['status'] == 'feasible' for x in r['final_deletion_checks'])
    assert r['solver_calls'] == len(r['checks']) == 7
    assert not any(r[k] for k in ('minimum_cardinality_claimed', 'unique_cause_claimed',
        'full_support_claimed', 'walking_claimed', 'standing_claimed', 'biological_validation'))
    for x, y in zip(args, saved): np.testing.assert_array_equal(x, y)


def test_singleton_conflict():
    args = list(problem()); args[2][1] = 2
    r = call(args)
    assert r['conflict_rows'] == [1] and r['inclusion_minimal']


def test_root_infeasibility_is_not_a_passive_cause():
    args = list(problem()); args[2][0] = -1
    r = call(args)
    assert r['status'] == 'root_balance_unresolved'
    assert not r['inclusion_minimal'] and r['solver_calls'] == 1


def test_no_conflict_does_not_certify_full_actuated_equations():
    args = list(problem()); args[2][2] = 0
    r = call(args)
    assert r['status'] == 'no_passive_conflict_found'
    assert not r['full_support_claimed']


def test_only_exact_zero_rows_are_passive():
    args = list(problem()); args[0][2, 0] = 1e-25
    r = call(args)
    assert r['passive_rows'] == [1]
    assert r['status'] == 'no_passive_conflict_found'


@pytest.mark.parametrize('key,value', [('wall_seconds', 0), ('wall_seconds', True),
    ('wall_seconds', float('nan')), ('wall_seconds', float('inf')), ('wall_seconds', 10**1000),
    ('max_calls', 0), ('max_calls', False), ('max_calls', 1.5)])
def test_invalid_budgets(key, value):
    with pytest.raises(ValueError): call(**{key: value})


@pytest.mark.parametrize('root_rows', [(), (0, 0), (True,), (-1,), (3,), (0.,)])
def test_invalid_root_selection(root_rows):
    with pytest.raises(ValueError): sc.diagnose(*problem(), root_rows=root_rows)


@pytest.mark.parametrize('index,mutation', [(0, 'nan'), (1, 'nan'), (2, 'nan'),
    (3, 'zero'), (4, 'negative'), (0, 'shape'), (1, 'shape'), (2, 'shape'),
    (3, 'shape'), (4, 'shape')])
def test_invalid_numeric_inputs(index, mutation):
    args = list(problem())
    if mutation == 'shape': args[index] = np.zeros(0)
    elif mutation == 'nan': args[index].flat[0] = np.nan
    elif mutation == 'zero': args[index].flat[0] = 0
    else: args[index].flat[0] = -1
    with pytest.raises(ValueError): call(args)


def test_call_limit_retains_completed_checks():
    with pytest.raises(TimeoutError) as caught: call(max_calls=1)
    assert caught.value.diagnostic['status'] == 'inconclusive'
    assert caught.value.diagnostic['solver_calls'] == 1
    assert not caught.value.diagnostic['inclusion_minimal']


@pytest.mark.parametrize('status', ['inconclusive', 'invalid_solver_solution', 'residual_check_failed'])
def test_solver_uncertainty_never_becomes_infeasibility(monkeypatch, status):
    monkeypatch.setattr(sc, 'solve_support', lambda *a: {'status': status, 'feasible': False})
    with pytest.raises(sc.UnresolvedSupport) as caught: call()
    assert len(caught.value.diagnostic['checks']) == 1


@pytest.mark.parametrize('key,value', [('feasible', 1), ('solver_status', True),
    ('solver_status', 2), ('solver_success', False)])
def test_inconsistent_solver_status_fails_closed(monkeypatch, key, value):
    result = {'status': 'feasible', 'feasible': True, 'solver_status': 0, 'solver_success': True}
    result[key] = value
    monkeypatch.setattr(sc, 'solve_support', lambda *a: result)
    with pytest.raises(ValueError): call()


def test_deadline_exceeded_inside_final_solver_is_reported(monkeypatch):
    clock = [0.]
    original = sc.solve_support
    monkeypatch.setattr(sc.time, 'monotonic', lambda: clock[0])
    def slow(*args):
        result = original(*args); clock[0] += 2.; return result
    monkeypatch.setattr(sc, 'solve_support', slow)
    with pytest.raises(TimeoutError) as caught: call(wall_seconds=1.)
    assert caught.value.diagnostic['solver_calls'] == 1
    assert caught.value.diagnostic['status'] == 'inconclusive'


def workspace(path):
    spec = importlib.util.spec_from_file_location('placement_fixture',
        Path(__file__).with_name('test_foot_placement_evidence.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.make_workspace(path)
    return path


def test_source_reverification_no_contacts_and_nonoverwrite(tmp_path):
    root = workspace(tmp_path)
    source = root/'foot-placement/observations.npz'; original = source.read_bytes()
    r = sc.run(root)
    assert r['status'] == 'completed' and r['candidates_checked'] == 0
    assert r['conflicts_isolated'] == 0 and r['physical_inputs_unchanged']
    assert source.read_bytes() == original
    result = root/'passive-conflict/result.json'; prior = result.read_bytes()
    with pytest.raises(FileExistsError): sc.run(root)
    assert result.read_bytes() == prior


def test_invalid_source_writes_failure_not_a_success(tmp_path):
    root = workspace(tmp_path)
    source = root/'foot-placement/result.json'
    record = json.loads(source.read_text()); record['walking_claimed'] = True
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError): sc.run(root)
    failure = json.loads((root/'passive-conflict/result.json').read_text())
    assert failure['status'] == 'failed'
    assert not failure['walking_claimed'] and not failure['candidates']


def test_inclusion_minimal_is_not_minimum_cardinality_or_unique_cause():
    T, C, b, limit, mu = problem()
    T = np.zeros((4, 1)); C = np.vstack((C[0], C[1], C[1], C[2]))
    # Row1 alone is already impossible, but deterministic deletion can discard
    # that explanation because independent rows2+3 still form a conflict.
    r = sc.diagnose(T, C, [1, 2, 1, 1], limit, mu, root_rows=(0,))
    assert r['single_rows'][0]['status'] == sc.INFEASIBLE
    assert r['conflict_rows'] == [2, 3] and r['inclusion_minimal']
    assert not r['minimum_cardinality_claimed'] and not r['unique_cause_claimed']


@pytest.mark.parametrize('bad_call', [5, 6])
def test_final_fresh_checks_must_reproduce(monkeypatch, bad_call):
    original = sc.solve_support; count = [0]
    def inconsistent(*args):
        count[0] += 1
        report = original(*args)
        if count[0] == bad_call:
            feasible = not report['feasible']
            return {'status': 'feasible' if feasible else sc.INFEASIBLE, 'feasible': feasible,
                    'solver_success': feasible, 'solver_status': 0 if feasible else 2}
        return report
    monkeypatch.setattr(sc, 'solve_support', inconsistent)
    with pytest.raises(sc.UnresolvedSupport) as caught: call()
    assert not caught.value.diagnostic['inclusion_minimal']
    assert caught.value.diagnostic['solver_calls'] == bad_call
