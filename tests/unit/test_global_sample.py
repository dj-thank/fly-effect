"""Protocol-level tests for the bounded global posture sampling module."""
import json
import numpy as np
import pytest

from organism_core import global_sample
from organism_core.global_sample import PROTOCOL, _check_descent_replay


def test_protocol_declares_bounded_search():
    assert PROTOCOL['id'] == 'FE-02-global-sample-v1'
    assert PROTOCOL['n_samples'] > 0
    assert PROTOCOL['rng_seed'] >= 0
    assert 'global_basin_improvement' in PROTOCOL['outcomes']
    assert 'no_global_improvement' in PROTOCOL['outcomes']
    assert 'tolerance_support_witness' in PROTOCOL['outcomes']
    assert 'inconclusive' in PROTOCOL['outcomes']


def test_example_record_matches_protocol():
    from pathlib import Path
    record = json.loads((Path(global_sample.__file__).resolve().parents[1]
                         / 'examples/fe02-global-sample.json').read_text(
                             encoding='utf-8'))
    assert record['parameters'] == PROTOCOL
    assert record['status'] == 'planned'


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


def test_check_replay_final_r_consistency():
    srec = {'descent': {'accepted': [{'accepted': 0, 'R': 0.1}],
                        'accepted_steps': 0, 'final_R': 0.2,
                        'terminal_qpos': [0.]*7, 'terminal_tilt_rad': 0.0}}
    und, errs = _check_descent_replay(0, srec, {})
    assert any('final_R differs' in e for e in errs)
