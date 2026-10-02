import copy
import numpy as np
import pytest
from organism_core.muscles import MuscleBank


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, 0., -1.])
def test_invalid_pool_denominators_are_rejected(bad):
    with pytest.raises(ValueError):
        MuscleBank([0.], [[bad, 1.]])


def test_pool_input_does_not_alias_live_state():
    pool = np.ones((1, 2))
    bank = MuscleBank([0.], pool)
    pool[:] = 0.
    np.testing.assert_array_equal(bank.pool_sizes, [[1., 1.]])


def assert_equal_state(left, right):
    assert left.keys() == right.keys()
    for name in left:
        if isinstance(left[name], np.ndarray):
            np.testing.assert_array_equal(left[name], right[name])
        else:
            assert left[name] == right[name]


@pytest.mark.parametrize('name', ['activation', 'rate', 'pending', 'fiber_force', 'last_torque'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_restore_is_rejected_atomically(name, value):
    bank = MuscleBank([0.]); before = bank.state(); bad = copy.deepcopy(before)
    bad['activation'][0, 0] = .5  # A valid early field must not be committed on later failure.
    bad[name].flat[0] = value
    with pytest.raises(ValueError):
        bank.restore(bad)
    assert_equal_state(before, bank.state())


@pytest.mark.parametrize('name', ['activation', 'rate', 'pending', 'fiber_force', 'saturated'])
def test_negative_state_rejected(name):
    bank = MuscleBank([0.]); bad = bank.state(); bad[name].flat[0] = -1
    with pytest.raises(ValueError):
        bank.restore(bad)


@pytest.mark.parametrize('value', [-1, 1.5, True, np.nan, np.inf, '1'])
def test_invalid_tick_rejected(value):
    bank = MuscleBank([0.]); bad = bank.state(); bad['tick'] = value
    with pytest.raises(ValueError):
        bank.restore(bad)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'dtype', 'shape', 'activation', 'work'])
def test_malformed_snapshot_rejected_without_mutation(mutation):
    bank = MuscleBank([0.]); before = bank.state(); bad = copy.deepcopy(before)
    bad['activation'][0, 0] = .5
    if mutation == 'missing': del bad['tick']
    elif mutation == 'extra': bad['surprise'] = 1
    elif mutation == 'dtype': bad['rate'] = bad['rate'].astype(np.float32)
    elif mutation == 'shape': bad['rate'] = np.zeros(2)
    elif mutation == 'activation': bad['activation'][0, 0] = 2.
    elif mutation == 'work': bad['work_native'] = float('nan')
    with pytest.raises(ValueError): bank.restore(bad)
    assert_equal_state(before, bank.state())


def test_signed_work_and_torque_are_valid_and_restore_has_no_alias():
    bank = MuscleBank([0.]); state = bank.state()
    state['last_torque'][0] = -1.; state['work_native'] = -3.; state['tick'] = 5
    bank.restore(state)
    assert bank.last_torque[0] == -1. and bank.work_native == -3.
    state['last_torque'][0] = 7.
    assert bank.last_torque[0] == -1.
