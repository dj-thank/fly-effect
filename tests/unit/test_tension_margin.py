"""Unit tests for the tension-authority margin (FE-02)."""
import numpy as np
import pytest
from organism_core.tension_margin import (PROTOCOL, protocol_hash,
                                          tension_margin_solve)

LIMITS = np.asarray([1., 1.])
MU = np.asarray([1.])


def synthetic(b6):
    """One contact, rows: rx=ry=0; f1+rz=1.5; f2=.4; passive row rz=b6."""
    T = np.zeros((7, 2)); C = np.zeros((7, 3)); b = np.zeros(7)
    C[0, 0] = C[1, 1] = 1.          # rx = 0, ry = 0
    T[3, 0] = C[3, 2] = 1.; b[3] = 1.5   # f1 + rz = 1.5
    T[4, 1] = 1.; b[4] = .4         # f2 = .4
    C[6, 2] = 1.; b[6] = b6         # contact-only row: rz = b6
    return T, C, b


def solve(b6, limits=LIMITS):
    T, C, b = synthetic(b6)
    return tension_margin_solve(T, C, b, limits, MU)


class TestTensionMarginSolve:
    def test_within_authority_boundary(self):
        report = solve(.5)            # f1 = 1.0, f2 = .4 -> s* = 1.0
        assert report['status'] == 'measured'
        assert report['s'] == pytest.approx(1., abs=1e-6)
        assert report['within_authority'] is True

    def test_bounded_deficit(self):
        report = solve(.2)            # f1 = 1.3 -> s* = 1.3
        assert report['status'] == 'measured'
        assert report['s'] == pytest.approx(1.3, abs=1e-6)
        assert report['within_authority'] is False
        assert report['biological_validation'] is False
        np.testing.assert_allclose(
            np.asarray(report['tensions_native'])/LIMITS, [1.3, .4],
            rtol=1e-6, atol=1e-6)

    def test_deficit_driven_by_second_tendon(self):
        report = solve(1.4)           # f1 = .1, f2 = .4 -> s* = .4
        assert report['status'] == 'measured'
        assert report['s'] == pytest.approx(.4, abs=1e-6)
        assert report['within_authority'] is True

    def test_unreachable_at_any_scale(self):
        report = solve(-.5)           # rz = -.5 impossible (rz >= 0)
        assert report['status'] == 'unreachable_at_any_scale'
        assert report['solver_status'] == 2

    def test_zero_needed_when_contact_covers(self):
        report = solve(1.5, limits=np.asarray([1., 1.]))
        # rz = 1.5 covers row 3 alone (f1 = 0); f2 = .4 still needs s = .4
        assert report['s'] == pytest.approx(.4, abs=1e-6)

    def test_rejects_inconsistent_dimensions(self):
        T, C, b = synthetic(.5)
        with pytest.raises(ValueError):
            tension_margin_solve(T, C, b, np.asarray([1.]), MU)
        with pytest.raises(ValueError):
            tension_margin_solve(T, C[:, :3], b[:6], LIMITS, MU)

    def test_rejects_nonpositive_limits(self):
        T, C, b = synthetic(.5)
        with pytest.raises(ValueError):
            tension_margin_solve(T, C, b, np.asarray([0., 1.]), MU)

    def test_solver_contract_fields(self):
        report = solve(.2)
        assert report['scope'] and 'fixed-geometry' in report['scope'].lower()
        for key in ('solver_status', 'solver_message', 'solver_success',
                    'maximum_scaled_residual', 'maximum_constraint_violation',
                    'tensions_native', 'contact_forces_native',
                    'tension_fraction_of_declared', 'saturated_tendons',
                    'solver_checks_passed'):
            assert key in report


class TestProtocol:
    def test_declared_outcomes_and_scope(self):
        assert PROTOCOL['id'] == 'FE-02-tension-margin-v1'
        assert PROTOCOL['dynamic_trials'] == 0
        assert PROTOCOL['biological_validation'] is False
        for outcome in ('within_tension_authority', 'bounded_tension_deficit',
                        'unbounded_tension_deficit', 'inconclusive',
                        'invalid_experiment'):
            assert outcome in PROTOCOL['outcomes']
        assert PROTOCOL['sources'] == ['foot-placement', 'passive-posture',
                                       'posture-shift', 'iterative-shift']

    def test_hash_stable(self):
        assert protocol_hash() == protocol_hash()
        assert len(protocol_hash()) == 64

    def test_versioned_record_matches(self):
        import json
        from pathlib import Path
        record = json.loads((Path(__file__).resolve().parents[2]
                             / 'examples/fe02-tension-margin.json').read_text())
        assert record['parameters'] == PROTOCOL
        assert record['biological_validation'] is False
        assert record['status'] == 'planned'
