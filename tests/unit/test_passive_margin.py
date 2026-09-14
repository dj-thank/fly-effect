"""Unit tests for the combined passive spring-authority margin (FE-02)."""
import numpy as np
import pytest
from organism_core.passive_margin import (INFEASIBLE, PROTOCOL,
                                          achievable_intervals, margin_solve,
                                          passive_rows, protocol_hash)

ROOTS = list(range(6))
LIMITS = np.asarray([1., 1.])
MU = np.asarray([1.])


def synthetic(b6, rows=7):
    """One contact, root rows fix (rx,ry,f1,f2,rz); passive row 6 needs rz."""
    T = np.zeros((rows, 2)); C = np.zeros((rows, 3)); b = np.zeros(rows)
    C[0, 0] = C[1, 1] = 1.          # rx = 0, ry = 0
    T[3, 0] = C[3, 2] = 1.; b[3] = 1.5   # f1 + rz = 1.5 -> f1 = 1, rz = .5
    T[4, 1] = 1.; b[4] = .4         # f2 = .4
    C[6, 2] = 1.; b[6] = b6         # passive row: residual = .5 - b6
    return T, C, b


def solve(b6, lo, hi):
    T, C, b = synthetic(b6)
    return margin_solve(T, C, b, LIMITS, MU, ROOTS, [6],
                        np.asarray([[lo, hi]]))


class TestIntervals:
    def test_values(self):
        I = achievable_intervals([2.], [[-1., 1.]], [.25])
        np.testing.assert_allclose(I, [[2.*(-1.-.25), 2.*(1.-.25)]])

    def test_zero_at_rest_inside(self):
        I = achievable_intervals([1., 3.], [[-.5, .5], [-1., 2.]], [.5, 1.])
        assert np.all(I[:, 0] <= 0.) and np.all(I[:, 1] >= 0.)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            achievable_intervals([1.], [[1., -1.]], [0.])
        with pytest.raises(ValueError):
            achievable_intervals([1.], [[-1., 1.]], [2.])  # q outside range
        with pytest.raises(ValueError):
            achievable_intervals([-1.], [[-1., 1.]], [0.])  # negative stiffness


class TestMarginSolve:
    def test_already_feasible_is_zero(self):
        report = solve(.5, -1., 1.)     # residual .5 - b6 = 0
        assert report['status'] == 'measured'
        assert report['t'] == pytest.approx(0., abs=1e-9)
        assert report['feasible_within_authority'] is True

    def test_within_combined_authority(self):
        report = solve(.2, -.4, .4)     # need +.3, interval half-width .4
        assert report['status'] == 'measured'
        assert report['t'] == pytest.approx(.75, abs=1e-6)
        assert report['feasible_within_authority'] is True
        assert report['required_shift_native'][0] == pytest.approx(.3, abs=1e-6)
        assert report['interval_respected'] is True

    def test_beyond_combined_authority(self):
        report = solve(.2, -.1, .1)     # need +.3, only .1 achievable
        assert report['status'] == 'measured'
        assert report['t'] == pytest.approx(3., abs=1e-6)
        assert report['feasible_within_authority'] is False
        assert report['biological_validation'] is False

    def test_wrong_sign_unreachable_at_any_shift(self):
        report = solve(.2, -.2, 0.)     # need +.3 but joint already at hi
        assert report['status'] == 'unreachable_at_any_shift'
        assert report['solver_status'] == 2

    def test_underside_direction(self):
        report = solve(1.9, -1., 1.)    # residual range [-1.4,-.4]; closest is -.4
        assert report['status'] == 'measured'
        assert report['t'] == pytest.approx(.4, abs=1e-6)
        assert report['required_shift_native'][0] == pytest.approx(-.4, abs=1e-6)

    def test_rejects_tendon_actuated_passive_row(self):
        T, C, b = synthetic(.5)
        T[6, 0] = 1.                  # row 6 is no longer passive
        with pytest.raises(ValueError):
            margin_solve(T, C, b, LIMITS, MU, ROOTS, [6], np.asarray([[-1., 1.]]))

    def test_rejects_inconsistent_dimensions(self):
        T, C, b = synthetic(.5)
        with pytest.raises(ValueError):
            margin_solve(T, C, b, LIMITS, MU, ROOTS, [6], np.asarray([[-1., 1.], [-1., 1.]]))

    def test_passive_rows_detection(self):
        T, _, _ = synthetic(.5)
        assert passive_rows(T, ROOTS) == [6]

    def test_solver_contract_fields(self):
        report = solve(.2, -.4, .4)
        assert report['scope'] and 'fixed-geometry' in report['scope'].lower()
        for key in ('solver_status', 'solver_message', 'solver_success',
                    'maximum_scaled_root_residual', 'maximum_constraint_violation',
                    'tensions_native', 'contact_forces_native',
                    'shift_fraction_of_interval', 'solver_checks_passed'):
            assert key in report


class TestProtocol:
    def test_declared_outcomes_and_scope(self):
        assert PROTOCOL['dynamic_trials'] == 0
        assert PROTOCOL['biological_validation'] is False
        assert 'within_combined_authority' in PROTOCOL['outcomes']
        assert 'beyond_combined_authority' in PROTOCOL['outcomes']
        assert 'inconclusive' in PROTOCOL['outcomes']

    def test_hash_stable(self):
        assert protocol_hash() == protocol_hash()
        assert len(protocol_hash()) == 64


def test_no_change_to_infeasible_label():
    assert INFEASIBLE == 'infeasible_under_declared_constraints'
