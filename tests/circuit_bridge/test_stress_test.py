import numpy as np
import pytest
from research.circuit_bridge.stress_test import resolve_ids, challenge_drive, GROUPS


def test_source_resolution_preserves_requested_channel_order():
    np.testing.assert_array_equal(resolve_ids(np.array([40,10,30]),[30,40]),[2,0])


@pytest.mark.parametrize('wanted',[[30,30],[99]])
def test_bad_sources_rejected(wanted):
    with pytest.raises(ValueError):resolve_ids(np.array([10,30]),wanted)


@pytest.mark.parametrize('name',['fast_alternating','long_silence_and_burst',*GROUPS])
def test_challenges_deterministic_finite_and_source_width(name):
    a=challenge_drive(name);b=challenge_drive(name)
    np.testing.assert_array_equal(a,b)
    assert a.shape == (160,len(GROUPS[name]) if name in GROUPS else 2)
    assert np.isfinite(a).all() and np.min(a)>=0 and np.max(a)<=1.1


def test_unknown_challenge_rejected():
    with pytest.raises(ValueError):challenge_drive('unknown')
