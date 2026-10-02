from types import SimpleNamespace
import numpy as np
import pytest
from organism_core.force_audit import require_unforced_static_state

FIELDS = ('qpos', 'qvel', 'qfrc_bias', 'qfrc_passive', 'qfrc_applied',
          'xfrc_applied', 'qfrc_actuator', 'ctrl', 'act')


def state():
    return SimpleNamespace(**{name: np.zeros(6) for name in FIELDS})


def test_unforced_static_contract_does_not_require_zero_gravity_or_passive_force():
    d = state(); d.qfrc_bias[:] = 7.; d.qfrc_passive[:] = .3; d.qpos[:] = 1.
    before = {name: getattr(d, name).copy() for name in FIELDS}
    require_unforced_static_state(d)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(d, name), value)


@pytest.mark.parametrize('name', FIELDS)
@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_force_contract_fails_before_solver(name, value):
    d = state(); getattr(d, name)[0] = value
    with pytest.raises(ValueError, match='finite'):
        require_unforced_static_state(d)


@pytest.mark.parametrize('name', ['qvel', 'qfrc_applied', 'xfrc_applied',
                                   'qfrc_actuator', 'ctrl', 'act'])
def test_hidden_motion_force_control_or_activation_fails_closed(name):
    d = state(); getattr(d, name)[0] = 1e-20
    with pytest.raises(ValueError, match='zero'):
        require_unforced_static_state(d)


def test_models_without_control_or_activation_are_supported():
    d = state(); d.ctrl = np.zeros(0); d.act = np.zeros(0)
    require_unforced_static_state(d)
