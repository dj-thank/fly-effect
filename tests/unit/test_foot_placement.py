"""Analytical and malformed-input controls, not biological validation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from organism_core.foot_placement import PROTOCOL, root_balance, select_sample, validate_sample


def wrench(points):
    blocks = []
    for x, y, z in points:
        blocks.append(np.vstack([np.eye(3), [[0, -z, y], [z, 0, -x], [-y, x, 0]]]))
    return np.column_stack(blocks)


def test_four_feet_support_centered_weight_and_force_witness():
    C = wrench([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]])
    b = np.array([0, 0, 10, 0, 0, 0.]); r = root_balance(C, b, np.full(4, .5))
    assert r['feasible'] and r['status'] == 'feasible'
    np.testing.assert_allclose(C@np.array(r['forces_native']).ravel(), b, atol=1e-10)


def test_two_hind_feet_cannot_balance_centered_weight():
    r = root_balance(wrench([[-1, -1, 0], [-1, 1, 0]]), [0, 0, 10, 0, 0, 0], [.5, .5])
    assert not r['feasible'] and r['status'] == 'infeasible_under_declared_constraints'


@pytest.mark.parametrize('horizontal,expected', [(0, True), (.5, True), (1.1, False)])
def test_fixed_friction_diamond_is_enforced(horizontal, expected):
    assert root_balance(wrench([[0, 0, 0]]), [horizontal, 0, 2, 0, 0, 0], [.5])['feasible'] == expected


@pytest.mark.parametrize('C,b,mu', [
    (np.zeros((5, 3)), np.zeros(6), [1]), (np.zeros((6, 0)), np.zeros(6), []),
    (np.zeros((6, 4)), np.zeros(6), [1]), (np.zeros((6, 3)), np.zeros(5), [1]),
    (np.zeros((6, 3)), np.zeros(6), []), (np.full((6, 3), np.nan), np.zeros(6), [1]),
    (np.zeros((6, 3)), np.full(6, np.inf), [1]), (np.zeros((6, 3)), np.zeros(6), [-1]),
    (np.zeros((6, 3)), np.zeros(6), [np.nan]),
])
def test_invalid_balance_data_are_not_infeasibility(C, b, mu):
    with pytest.raises(ValueError): root_balance(C, b, mu)


@pytest.mark.parametrize('solution,expected', [
    (SimpleNamespace(status=1, success=False), 'inconclusive'),
    (SimpleNamespace(status=0, success=True, x=[np.nan, 0, 0]), 'invalid_solver_solution'),
    (SimpleNamespace(status=0, success=True, x=[0, 0]), 'invalid_solver_solution'),
    (SimpleNamespace(status=0, success=True, x=[0, 0, 0]), 'residual_check_failed'),
])
def test_solver_failures_not_reported_as_success(monkeypatch, solution, expected):
    monkeypatch.setattr('scipy.optimize.linprog', lambda *a, **kw: solution)
    r = root_balance(wrench([[0, 0, 0]]), [0, 0, 1, 0, 0, 0], [.5])
    assert not r['feasible'] and r['status'] == expected


def sample(index=0, feet=('lf',), feasible=False, depth=.001, nonfoot=()):
    return {'index': index, 'foot_legs': list(feet), 'nonfoot_contacts': list(nonfoot),
            'depth_native': depth, 'root_balance': {'feasible': feasible}}


def test_depth_selection_priorities_and_nonfoot_exclusion():
    rows = [sample(0, ('lf', 'rf'), False), sample(1, ('lf',), True, .02),
            sample(2, ('lf', 'rf'), True, .01), sample(3, ('lf', 'rf'), True, .005),
            sample(4, tuple('abcdef'), True, .001, ['thorax'])]
    assert select_sample(rows)['index'] == 3
    assert select_sample([sample(feet=())]) is None
    assert select_sample([]) is None


def contact_sample():
    return {'foot_legs': ['lf'], 'nonfoot_contacts': [],
            'contacts': [{'geom': 'organism/lf_tarsus5', 'mu': .5, 'efc_address': 0}],
            'root_balance': {'feasible': False, 'status': 'infeasible_under_declared_constraints'}}


@pytest.mark.parametrize('mutation', ['feet', 'friction', 'active', 'nonfoot', 'columns'])
def test_contact_metadata_cannot_forge_eligibility(mutation):
    s = contact_sample(); C = np.zeros((6, 3)); mu = [.5]
    if mutation == 'feet': s['foot_legs'] = ['lf', 'rf']
    elif mutation == 'friction': mu = [.8]
    elif mutation == 'active': s['contacts'][0]['efc_address'] = -1
    elif mutation == 'nonfoot': s['nonfoot_contacts'] = ['thorax']
    else: C = np.zeros((6, 6))
    with pytest.raises(ValueError): validate_sample(s, C, mu)


def test_valid_contact_metadata():
    validate_sample(contact_sample(), np.zeros((6, 3)), [.5])


def test_versioned_protocol_is_exact_and_has_no_dynamic_claims():
    p = Path(__file__).resolve().parents[2]/'examples/fe02-foot-placement.json'
    record = json.loads(p.read_text())
    assert record['parameters'] == PROTOCOL
    assert PROTOCOL['candidate_count'] == 1+len(PROTOCOL['xy_scales'])*len(PROTOCOL['target_z_offsets_native'])
    assert PROTOCOL['dynamic_trials'] == 0 and record['biological_validation'] is False
