"""Protocol-level tests for the full-system residual descent module."""
import json
import numpy as np
import pytest

from organism_core import full_descent
from organism_core.full_descent import (PROTOCOL, _check_descent_replay,
                                        _full_residual)


def test_protocol_declares_bounded_search():
    assert PROTOCOL['id'] == 'FE-02-full-descent-v1'
    assert PROTOCOL['support_residual_tolerance'] > 0
    assert 'tolerance_support_witness' in PROTOCOL['outcomes']
    assert 'inconclusive' in PROTOCOL['outcomes']
    assert 'invalid_experiment' in PROTOCOL['outcomes']


def test_example_record_matches_protocol():
    from pathlib import Path
    record = json.loads((Path(full_descent.__file__).resolve().parents[1]
                         / 'examples/fe02-full-descent.json').read_text(
                             encoding='utf-8'))
    assert record['parameters'] == PROTOCOL
    assert record['status'] == 'planned'


def test_full_residual_feasible_is_zero():
    R, st = _full_residual({'status': 'feasible'})
    assert R == 0. and st == 'feasible'


def test_full_residual_infeasible_uses_nearest():
    R, st = _full_residual({'status': 'infeasible_under_declared_constraints',
                            'nearest_balance': {
                                'minimum_scaled_balance_error': 0.21}})
    assert R == pytest.approx(0.21) and st == 'infeasible'


def test_full_residual_infeasible_without_nearest_is_inf():
    R, st = _full_residual({'status': 'infeasible_under_declared_constraints'})
    assert not np.isfinite(R)


def test_full_residual_check_failed_uses_achieved_error():
    R, st = _full_residual({'status': 'residual_check_failed',
                            'maximum_scaled_residual': 2e-6})
    assert R == pytest.approx(2e-6) and st == 'residual_check_failed'


def test_full_residual_inconclusive_is_inf():
    R, st = _full_residual({'status': 'inconclusive'})
    assert not np.isfinite(R)


def test_check_replay_flags_missing_inputs():
    srec = {'descent': {'accepted': [{'accepted': 2, 'R': 0.1}],
                        'accepted_steps': 2, 'final_R': 0.1,
                        'terminal_qpos': [0.]*7, 'terminal_tilt_rad': 0.1}}
    und, errs = _check_descent_replay(0, srec, {})
    assert any('saved no inputs' in e for e in errs)


def test_check_replay_rejects_tilt_over_bound():
    srec = {'descent': {'accepted': [{'accepted': 0, 'R': 0.1}],
                        'accepted_steps': 0, 'final_R': 0.1,
                        'terminal_qpos': [0.]*7,
                        'terminal_tilt_rad': PROTOCOL['max_root_tilt_rad']+1.}}
    und, errs = _check_descent_replay(0, srec, {})
    assert any('tilt' in e for e in errs)
