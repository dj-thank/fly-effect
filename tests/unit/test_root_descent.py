"""Protocol-level tests for the root-orientation descent module."""
import json
import numpy as np
import pytest

from organism_core import root_descent
from organism_core.root_descent import PROTOCOL, _check_descent_replay, _tilt


def test_protocol_declares_bounded_search():
    assert PROTOCOL['id'] == 'FE-02-root-descent-v1'
    assert PROTOCOL['max_root_tilt_rad'] > 0
    assert PROTOCOL['max_evals_per_seed'] > 0
    assert 'inconclusive' in PROTOCOL['outcomes']
    assert 'invalid_experiment' in PROTOCOL['outcomes']


def test_example_record_matches_protocol():
    from pathlib import Path
    record = json.loads((Path(root_descent.__file__).resolve().parents[1]
                         / 'examples/fe02-root-descent.json').read_text(
                             encoding='utf-8'))
    assert record['parameters'] == PROTOCOL
    assert record['status'] == 'planned'


def test_tilt_is_zero_at_seed():
    q = np.zeros(7)
    q[3] = 1.0  # unit quat w,x,y,z
    assert _tilt(q, q) == pytest.approx(0.)


def test_tilt_measures_rotation_angle():
    q_seed = np.array([0., 0., 0., 1., 0., 0., 0.])
    angle = 0.3
    q = q_seed.copy()
    q[3] = np.cos(angle/2)
    q[4] = np.sin(angle/2)  # rotation about x
    assert _tilt(q, q_seed) == pytest.approx(angle)


def test_tilt_is_sign_invariant():
    q_seed = np.array([0., 0., 0., 1., 0., 0., 0.])
    q = q_seed.copy()
    q[3:7] = -q_seed[3:7]  # same physical orientation
    assert _tilt(q, q_seed) == pytest.approx(0.)


def test_check_replay_rejects_tilt_over_bound():
    srec = {'descent': {'accepted': [{'accepted': 0, 'R': 1e-7}],
                        'accepted_steps': 0, 'final_R': 1e-7,
                        'terminal_qpos': list(np.zeros(7)),
                        'terminal_tilt_rad': PROTOCOL['max_root_tilt_rad']+0.1}}
    und, errs = _check_descent_replay(0, srec, {}, [])
    assert any('tilt' in e for e in errs)


def test_check_replay_flags_missing_inputs():
    srec = {'descent': {'accepted': [{'accepted': 3, 'R': 5e-7}],
                        'accepted_steps': 3, 'final_R': 5e-7,
                        'terminal_qpos': [0.]*7, 'terminal_tilt_rad': 0.2}}
    und, errs = _check_descent_replay(0, srec, {}, [])
    assert any('saved no inputs' in e for e in errs)


def test_check_replay_final_r_consistency():
    srec = {'descent': {'accepted': [{'accepted': 0, 'R': 1e-7}],
                        'accepted_steps': 0, 'final_R': 9e-7,
                        'terminal_qpos': [0.]*7, 'terminal_tilt_rad': 0.0}}
    und, errs = _check_descent_replay(0, srec, {}, [])
    assert any('final_R differs' in e for e in errs)
