"""Unit tests for the bounded iterative posture-shift experiment (FE-02)."""
import json
import pytest
from organism_core.iterative_shift import (
    PROTOCOL, _check_terminal_iterate, protocol_hash)


def _sample(root=True, margin='measured'):
    return {'root_balance': {'feasible': root},
            'margin': {'status': margin},
            'passive_subsystem': {'status': 'infeasible'},
            'support': {'status': 'infeasible'}}


def _irec(driver=0, t=0.5, passive=False, full=False):
    rec = {'best_gate': 2, 'any_passive_feasible': passive,
           'any_full_feasible': full}
    if driver is not None:
        rec['driver_sample'] = driver
        rec['driver_margin_t'] = t
    return rec


class TestProtocol:
    def test_declared_outcomes_and_scope(self):
        assert PROTOCOL['id'] == 'FE-02-iterative-shift-v1'
        assert PROTOCOL['dynamic_trials'] == 0
        assert PROTOCOL['biological_validation'] is False
        for outcome in ('iterated_shift_reached_full_support',
                        'iterated_shift_reached_passive_subsystem',
                        'no_convergence_in_bounded_iteration',
                        'inconclusive', 'invalid_experiment'):
            assert outcome in PROTOCOL['outcomes']
        assert PROTOCOL['gates'] == ['foot_only_contact', 'root_balance',
                                     'passive_subsystem', 'full_support']
        assert PROTOCOL['max_iterations'] == 6
        assert 'no_root_feasible_sample' in PROTOCOL['stop']
        assert 'beyond_authority' in PROTOCOL['stop']
        assert 'iteration_cap' in PROTOCOL['stop']

    def test_budgets_declared(self):
        for key in ('max_solver_calls_per_candidate',
                    'max_saved_driver_sets_per_candidate',
                    'candidate_wall_s', 'worker_wall_s', 'memory_mib'):
            assert PROTOCOL[key] > 0

    def test_hash_stable(self):
        assert protocol_hash() == protocol_hash()
        assert len(protocol_hash()) == 64

    def test_versioned_record_matches(self):
        from pathlib import Path
        record = json.loads((Path(__file__).resolve().parents[2]
                             / 'examples/fe02-iterative-shift.json').read_text())
        assert record['parameters'] == PROTOCOL
        assert record['biological_validation'] is False
        assert record['status'] == 'planned'


class TestTerminalIterateCheck:
    def test_iteration_cap_last_iterate_accepted(self):
        # The final record under the cap is the terminal record even though
        # it still carries a valid driver (the cap stopped the loop).
        _check_terminal_iterate('iteration_cap', _irec(), [_sample()],
                                is_last=True)

    def test_nonterminal_iterate_requires_within_authority_driver(self):
        _check_terminal_iterate('iteration_cap', _irec(t=1.0), [_sample()],
                                is_last=False)

    def test_nonterminal_without_driver_rejected(self):
        with pytest.raises(ValueError, match='within-authority driver'):
            _check_terminal_iterate('iteration_cap', _irec(driver=None),
                                    [_sample()], is_last=False)

    def test_nonterminal_beyond_authority_driver_rejected(self):
        with pytest.raises(ValueError, match='within-authority driver'):
            _check_terminal_iterate('iteration_cap', _irec(t=1.01),
                                    [_sample()], is_last=False)

    def test_no_root_feasible_terminal(self):
        _check_terminal_iterate('no_root_feasible_sample',
                                _irec(driver=None), [_sample(root=False)],
                                is_last=True)

    def test_no_root_feasible_with_root_sample_rejected(self):
        with pytest.raises(ValueError, match='root-feasible sample exists'):
            _check_terminal_iterate('no_root_feasible_sample',
                                    _irec(driver=None), [_sample(root=True)],
                                    is_last=True)

    def test_beyond_authority_terminal(self):
        _check_terminal_iterate('beyond_authority', _irec(t=1.5),
                                [_sample()], is_last=True)

    def test_beyond_authority_inside_rejected(self):
        with pytest.raises(ValueError, match='inside authority'):
            _check_terminal_iterate('beyond_authority', _irec(t=0.9),
                                    [_sample()], is_last=True)

    def test_reached_passive_terminal(self):
        _check_terminal_iterate('reached_passive_subsystem',
                                _irec(passive=True), [_sample()], is_last=True)

    def test_reached_passive_without_sample_rejected(self):
        with pytest.raises(ValueError, match='passive status'):
            _check_terminal_iterate('reached_passive_subsystem', _irec(),
                                    [_sample()], is_last=True)

    def test_reached_full_terminal(self):
        _check_terminal_iterate('reached_full_support', _irec(full=True),
                                [_sample()], is_last=True)

    def test_reached_full_without_sample_rejected(self):
        with pytest.raises(ValueError, match='full-support status'):
            _check_terminal_iterate('reached_full_support', _irec(),
                                    [_sample()], is_last=True)

    def test_margin_unresolved_terminal(self):
        _check_terminal_iterate('margin_unresolved', _irec(driver=None),
                                [_sample(margin='inconclusive')], is_last=True)

    def test_margin_unresolved_with_measured_driver_rejected(self):
        with pytest.raises(ValueError, match='measured driver'):
            _check_terminal_iterate('margin_unresolved', _irec(driver=None),
                                    [_sample(margin='measured')],
                                    is_last=True)

    def test_margin_unresolved_without_root_sample_rejected(self):
        with pytest.raises(ValueError, match='measured driver'):
            _check_terminal_iterate('margin_unresolved', _irec(driver=None),
                                    [_sample(root=False)], is_last=True)

    def test_left_joint_limits_terminal_accepted(self):
        # The move after the last recorded iterate left the joint limits;
        # the last recorded pose itself needs no further justification.
        _check_terminal_iterate('left_joint_limits', _irec(), [_sample()],
                                is_last=True)
