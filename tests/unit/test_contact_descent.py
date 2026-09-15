"""Unit tests for live-contact residual descent (FE-02)."""
import json
from pathlib import Path
import numpy as np
import pytest
from organism_core.contact_descent import (PROTOCOL, WITNESS,
                                           protocol_hash)


class TestWitnessRegex:
    def test_matches_certification_prefix(self):
        m = WITNESS.match('cdesc_03_cert_01_depth_04_support_tendon_map')
        assert m and m.group(1) == 'cdesc_03_cert_01_depth_04_'

    def test_rejects_trajectory_and_foreign_keys(self):
        assert WITNESS.match('cdesc_00_acc_0007_support_target') is None
        assert WITNESS.match('cdesc_00_cert_00_depth_00_qpos') is None
        assert WITNESS.match(
            'desc_00_cert_00_depth_00_support_tendon_map') is None
        assert WITNESS.match(
            'pose_01_lineage_00_it_02_depth_03_support_tendon_map') is None


class TestProtocol:
    def test_declared_outcomes_and_scope(self):
        assert PROTOCOL['id'] == 'FE-02-contact-descent-v1'
        for outcome in ('found_full_support_pose',
                        'found_subsystem_feasible_pose',
                        'descent_floored_without_feasibility',
                        'inconclusive', 'invalid_experiment'):
            assert outcome in PROTOCOL['outcomes']
        assert 'LIVE' in PROTOCOL['objective']
        assert PROTOCOL['seed_margin_threshold'] == 1e-3
        assert PROTOCOL['h_rel_initial'] > PROTOCOL['h_rel_floor'] > 0

    def test_live_contact_contract(self):
        # The defining difference from residual-descent: contacts re-derived.
        assert 'recomputed' in PROTOCOL['objective']
        assert PROTOCOL['depths_native'] == [.001, .005, .01, .02, .04]

    def test_hash_stable(self):
        assert protocol_hash() == protocol_hash()
        assert len(protocol_hash()) == 64

    def test_versioned_record_matches(self):
        record = json.loads((Path(__file__).resolve().parents[2]
                             / 'examples/fe02-contact-descent.json').read_text())
        assert record['parameters'] == PROTOCOL
        assert record['biological_validation'] is False
        assert record['status'] == 'planned'
