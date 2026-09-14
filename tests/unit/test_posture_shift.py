"""Unit tests for the directed posture-shift experiment surface (FE-02)."""
import json
import pytest
from organism_core.posture_shift import PROTOCOL, protocol_hash


class TestProtocol:
    def test_declared_outcomes_and_scope(self):
        assert PROTOCOL['id'] == 'FE-02-posture-shift-v1'
        assert PROTOCOL['dynamic_trials'] == 0
        assert PROTOCOL['biological_validation'] is False
        for outcome in ('shift_reached_full_support',
                        'shift_reached_passive_subsystem',
                        'no_shift_reached_equilibrium', 'inconclusive'):
            assert outcome in PROTOCOL['outcomes']
        assert PROTOCOL['gates'] == ['foot_only_contact', 'root_balance',
                                     'passive_subsystem', 'full_support']
        assert PROTOCOL['depths_native'] == [.001, .005, .01, .02, .04]

    def test_budgets_declared(self):
        for key in ('max_solver_calls_per_candidate', 'candidate_wall_s',
                    'worker_wall_s', 'memory_mib'):
            assert PROTOCOL[key] > 0

    def test_hash_stable(self):
        assert protocol_hash() == protocol_hash()
        assert len(protocol_hash()) == 64

    def test_versioned_record_matches(self):
        from pathlib import Path
        record = json.loads((Path(__file__).resolve().parents[2]
                             / 'examples/fe02-posture-shift.json').read_text())
        assert record['parameters'] == PROTOCOL
        assert record['biological_validation'] is False
        assert record['status'] == 'planned'
