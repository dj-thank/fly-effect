"""Unit tests for residual-descent posture generation (FE-02)."""
import json
from pathlib import Path
import numpy as np
import pytest
from organism_core.residual_descent import (PROTOCOL, WITNESS, _residual,
                                            protocol_hash)

LIMITS = np.asarray([1., 1.])
MU = np.asarray([1.])


def synthetic(b6):
    """One contact, rows: rx=ry=0; f1+rz=1.5; f2=.4; contact-only row rz=b6."""
    T = np.zeros((7, 2)); C = np.zeros((7, 3)); b = np.zeros(7)
    C[0, 0] = C[1, 1] = 1.          # rx = 0, ry = 0
    T[3, 0] = C[3, 2] = 1.; b[3] = 1.5   # f1 + rz = 1.5
    T[4, 1] = 1.; b[4] = .4         # f2 = .4
    C[6, 2] = 1.; b[6] = b6         # contact-only row: rz = b6
    return T, C, b


class TestResidual:
    def test_feasible_rowset_gives_zero(self):
        T, C, b = synthetic(.5)
        R, st = _residual(T, C, b, LIMITS, MU, [3, 4, 6])
        assert st == 'feasible'
        assert R == 0.

    def test_infeasible_rowset_gives_finite_residual(self):
        T, C, b = synthetic(-.5)     # rz = -.5 unreachable (rz >= 0)
        R, st = _residual(T, C, b, LIMITS, MU, [3, 4, 6])
        assert st == 'infeasible'
        assert np.isfinite(R) and R > 0.

    def test_row_selection_changes_verdict(self):
        T, C, b = synthetic(-.5)
        R_ok, _ = _residual(T, C, b, LIMITS, MU, [3, 4])
        assert R_ok == 0.
        R_bad, st_bad = _residual(T, C, b, LIMITS, MU, [3, 4, 6])
        assert st_bad == 'infeasible' and R_bad > 0.

    def test_tendon_deficit_row_gives_positive_residual(self):
        T3, C3, b3 = synthetic(.5)
        C3[3, 2] = 0.; b3[3] = 1.5   # row 3: f1 alone = 1.5 (over limit)
        R, st = _residual(T3, C3, b3, LIMITS, MU, [3, 4, 6])
        assert st == 'infeasible'
        assert np.isfinite(R) and R > 0.


class TestWitnessRegex:
    def test_matches_certification_prefix(self):
        m = WITNESS.match('desc_03_cert_01_depth_04_support_tendon_map')
        assert m and m.group(1) == 'desc_03_cert_01_depth_04_'

    def test_rejects_trajectory_and_foreign_keys(self):
        assert WITNESS.match('desc_00_acc_0007_target') is None
        assert WITNESS.match('desc_00_cert_00_depth_00_qpos') is None
        assert WITNESS.match(
            'pose_01_lineage_00_it_02_depth_03_support_tendon_map') is None


class TestProtocol:
    def test_declared_outcomes_and_scope(self):
        assert PROTOCOL['id'] == 'FE-02-residual-descent-v1'
        for outcome in ('found_full_support_pose',
                        'found_subsystem_feasible_pose',
                        'descent_floored_without_feasibility',
                        'inconclusive', 'invalid_experiment'):
            assert outcome in PROTOCOL['outcomes']
        assert PROTOCOL['seed_margin_threshold'] == 1e-3
        assert PROTOCOL['h_rel_initial'] > PROTOCOL['h_rel_floor'] > 0
        assert PROTOCOL['max_evals_per_seed'] > 0

    def test_hash_stable(self):
        assert protocol_hash() == protocol_hash()
        assert len(protocol_hash()) == 64

    def test_versioned_record_matches(self):
        record = json.loads((Path(__file__).resolve().parents[2]
                             / 'examples/fe02-residual-descent.json').read_text())
        assert record['parameters'] == PROTOCOL
        assert record['biological_validation'] is False
        assert record['status'] == 'planned'
