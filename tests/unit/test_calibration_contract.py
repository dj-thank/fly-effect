import sys
import types
import json
import numpy as np
import pytest
from organism_core.calibration import run_body_probe
from fly_effect import doctor


@pytest.mark.parametrize('duration', [0., -.01, .1001, .00015, np.nan, np.inf, True])
def test_invalid_duration_refused_before_creating_output(tmp_path, duration):
    path = tmp_path / 'result'
    with pytest.raises(ValueError): run_body_probe(path, duration=duration)
    assert not path.exists()


@pytest.mark.parametrize('limit', [0., -1., 601., np.nan, np.inf])
def test_invalid_wall_budget_refused(tmp_path, limit):
    with pytest.raises(ValueError): run_body_probe(tmp_path / 'result', wall_limit=limit)


def test_missing_body_dependency_emits_failed_not_success_receipt(tmp_path, monkeypatch):
    fake = types.ModuleType('organism_core.body')
    class UnavailableBody:
        def __init__(self, *args, **kwargs): raise FileNotFoundError('fixture missing mesh')
    fake.Body = UnavailableBody; fake.UNIT_CONTRACT = {}
    monkeypatch.setitem(sys.modules, 'organism_core.body', fake)
    monkeypatch.setitem(sys.modules, 'mujoco', types.ModuleType('mujoco'))
    out = tmp_path / 'failed'
    with pytest.raises(FileNotFoundError, match='fixture missing mesh'): run_body_probe(out)
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'failed' and result['error_type'] == 'FileNotFoundError'
    assert not result['biological_validation'] and not result['walking_claimed']
    assert result['functional_gates_automatically_passed'] == []
    assert not (out / 'observation.npz').exists()


def test_doctor_discloses_body_python_requirement():
    info = doctor()['runtime_compatibility']
    assert info['body_python_supported'] == ((3, 12) <= sys.version_info[:2] < (3, 15))
    assert '3.12' in info['body_python_requirement']
