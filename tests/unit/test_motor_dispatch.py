"""Dispatch equality checks using synthetic indices, no neural dataset changes."""
import numpy as np
from organism_core.motor_map import MotorMap


def fixture():
    obj = MotorMap.__new__(MotorMap)
    obj.motor_set = set(range(100)); obj.assignments = {i: (i % 3, i % 2) for i in range(30)}
    obj.rows = [{'internal_index': i, 'required_for_walking': i % 2 == 0} for i in range(100)]
    obj._required_by_index = {r['internal_index']: r['required_for_walking'] for r in obj.rows}
    obj.pool_sizes = np.ones((3, 2)); obj.outside_spikes = obj.unresolved_spikes = obj.mapped_spikes = 0
    obj.per_motor_spikes = dict.fromkeys(obj.motor_set, 0)
    return obj


def reference(obj, fired):
    counts = np.zeros_like(obj.pool_sizes)
    for index in map(int, fired):
        if index not in obj.motor_set: continue
        obj.per_motor_spikes[index] += 1
        if index in obj.assignments:
            j, pol = obj.assignments[index]; counts[j, pol] += 1; obj.mapped_spikes += 1
        else:
            row = next(r for r in obj.rows if r['internal_index'] == index)
            if row['required_for_walking']: obj.unresolved_spikes += 1
            else: obj.outside_spikes += 1
    return counts


def test_dispatch_matches_original_counts_and_accounting_over_repeated_ticks():
    a, b = fixture(), fixture(); rng = np.random.default_rng(17)
    for _ in range(100):
        fired = rng.integers(-5, 130, 200)
        np.testing.assert_array_equal(a.decode(fired), reference(b, fired))
        assert a.state() == b.state()
    restored = fixture(); restored.restore(a.state())
    assert restored.state() == a.state()


def test_decode_does_not_walk_annotation_rows():
    obj = fixture()
    class ForbiddenRows:
        def __iter__(self): raise AssertionError('Per-spike annotation scan regressed')
    obj.rows = ForbiddenRows()
    counts = obj.decode([1, 1, 98, 99, 1000])
    assert counts.sum() == 2
    assert obj.mapped_spikes == 2 and obj.unresolved_spikes == obj.outside_spikes == 1


def test_empty_spike_batch_leaves_state_unchanged():
    obj = fixture(); before = obj.state()
    assert not obj.decode([]).any()
    assert obj.state() == before
