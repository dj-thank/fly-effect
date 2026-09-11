"""Small synthetic graphs only: no claim about the unavailable MaleCNS payload."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
from organism_core import connectivity_audit as ca


def fixture_graph():
    # 0 -> 1 -> 3 -> 4(motor); 2 -> 1 is sign-zero, 5 is isolated.
    ids = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
    motors = np.array([4], dtype=np.int32)
    signs = np.array([1, -1, 0, 1, 0, 1], dtype=np.int8)
    edges = np.array([(0, 1, 2, 0), (1, 3, 4, 1), (3, 4, 7, 2),
                      (2, 1, 5, 3), (4, 0, 3, 4), (0, 1, 6, 5),
                      (3, 3, 8, 6), (5, 4, 0, 7)], dtype=ca.EDGE_DTYPE)
    return ids, motors, signs, edges


def slices(edges, step=2):
    return (edges[i:i + step] for i in range(0, len(edges), step))


def disk_fixture(path):
    path.mkdir()
    ids, motors, signs, edges = fixture_graph()
    np.save(path/'body_ids.npy', ids)
    np.save(path/'motor_indices.npy', motors)
    np.save(path/ca.SIGN_FILE, signs)
    edges.tofile(path/'edges.bin')
    (path/'neurons.parquet').write_bytes(b'synthetic unparsed annotation fixture')
    lock = {'dataset': 'synthetic-test-only', 'neurons': len(ids), 'edges': len(edges),
            'files': {name: hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ca.FILES}}
    return lock


def test_counts_are_not_percent_connected_or_unique_pairs():
    ids, motor, signs, edges = fixture_graph()
    result = ca.audit_arrays(ids, motor, signs, slices(edges))
    assert result['registered']['nonmotor'] == ca.ratio(5, 6)
    assert result['stored'] == {'edge_rows': 8, 'stored_synapse_count_sum': 35,
                                'zero_count_rows': 1, 'self_edge_rows': 1, 'unique_endpoint_pairs': None}
    structural = result['layers']['structural']
    effective = result['layers']['nonzero_model_weight']
    assert structural['edge_rows'] == 7
    assert effective['edge_rows'] == 5
    assert structural['groups']['nonmotor']['incident'] == ca.ratio(4, 5)
    assert effective['groups']['nonmotor']['incident'] == ca.ratio(3, 5)
    assert structural['groups']['nonmotor']['directly_targets_motor'] == ca.ratio(1, 5)
    assert structural['blocks']['nonmotor_to_nonmotor'] == {'edge_rows': 5, 'stored_synapse_count_sum': 25}
    assert structural['blocks']['nonmotor_to_motor']['stored_synapse_count_sum'] == 7
    assert structural['blocks']['motor_to_nonmotor']['edge_rows'] == 1
    assert effective['blocks']['motor_to_nonmotor']['edge_rows'] == 0
    assert result['reachability']['status'] == 'not_measured'
    assert result['biological_validation'] is False


@pytest.mark.parametrize('step', [1, 2, 3, 8, 100])
@pytest.mark.parametrize('threshold', [1, 5, 9])
def test_chunk_order_does_not_change_census(step, threshold):
    ids, motor, signs, edges = fixture_graph()
    expected = ca.audit_arrays(ids, motor, signs, [edges], min_count=threshold)
    assert ca.audit_arrays(ids, motor, signs, slices(edges[::-1].copy(), step), min_count=threshold) == expected


def test_self_only_and_empty_graph_denominators():
    result = ca.audit_arrays(np.array([1, 2]), np.array([], dtype=int), np.array([1, 0]),
                            [np.array([(0, 0, 1, 0)], dtype=ca.EDGE_DTYPE)])
    group = result['layers']['structural']['groups']
    assert group['all']['incident'] == ca.ratio(1, 2)
    assert group['all']['connected_to_other_neurons'] == ca.ratio(0, 2)
    assert group['motor']['incident']['percent'] is None
    empty = ca.audit_arrays(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), [])
    assert empty['registered']['motor']['percent'] is None


@pytest.mark.parametrize('threshold', [1, 5])
def test_reachability_is_directed_and_model_specific(threshold):
    ids, motor, signs, edges = fixture_graph()
    result = ca.motor_reachability(ids, motor, signs, lambda: slices(edges), min_count=threshold)
    assert result['status'] == 'exact'
    assert result['layers']['structural']['nonmotor_reaching_any_motor'] == ca.ratio(4 if threshold == 1 else 1, 5)
    assert result['layers']['nonzero_model_weight']['nonmotor_reaching_any_motor'] == ca.ratio(3 if threshold == 1 else 1, 5)


@pytest.mark.parametrize('step', [1, 2, 8])
def test_bounded_traversal_is_order_independent_lower_bound(step):
    ids, motor, signs, edges = fixture_graph()
    a = ca.motor_reachability(ids, motor, signs, lambda: slices(edges, step), max_passes=1)
    b = ca.motor_reachability(ids, motor, signs, lambda: slices(edges[::-1], step), max_passes=1)
    assert a == b
    assert a['status'] == 'lower_bound'
    assert a['layers']['structural']['nonmotor_reaching_any_motor'] == ca.ratio(1, 5)


def test_random_graphs_against_independent_python_set_oracle():
    rng = np.random.default_rng(240911)
    for _ in range(100):
        n = int(rng.integers(1, 15)); size = int(rng.integers(0, 80))
        ids = np.arange(n, dtype=np.int64) + 2**60
        motor = np.flatnonzero(rng.random(n) < .2)
        signs = rng.integers(-1, 2, n, dtype=np.int8)
        edges = np.zeros(size, dtype=ca.EDGE_DTYPE)
        edges['pre'] = rng.integers(n, size=size); edges['post'] = rng.integers(n, size=size)
        edges['count'] = rng.integers(0, 10, size=size)
        threshold = int(rng.integers(1, 6))
        actual = ca.audit_arrays(ids, motor, signs, slices(edges, 7), min_count=threshold)
        reachable = ca.motor_reachability(ids, motor, signs, lambda: slices(edges, 3), min_count=threshold)
        motors = set(map(int, motor)); nonmotor = set(range(n)) - motors
        for name in ca.LAYERS:
            selected = [(int(r['pre']), int(r['post']), int(r['count'])) for r in edges
                        if r['count'] >= threshold and (name == 'structural' or signs[r['pre']] != 0)]
            incident = {i for a, b, c in selected for i in (a, b)}
            assert actual['layers'][name]['groups']['nonmotor']['incident'] == ca.ratio(len(incident & nonmotor), len(nonmotor))
            for bi, key in enumerate(ca.BLOCKS):
                pairs = [(a, b, c) for a, b, c in selected if 2 * (a in motors) + (b in motors) == bi]
                assert actual['layers'][name]['blocks'][key] == {'edge_rows': len(pairs), 'stored_synapse_count_sum': sum(c for a, b, c in pairs)}
            # Independent adjacency-list reverse BFS, not repeated numpy scans.
            incoming = {i: set() for i in range(n)}
            for a, b, c in selected: incoming[b].add(a)
            seen = set(motors); queue = list(motors)
            while queue:
                for previous in incoming[queue.pop()] - seen:
                    seen.add(previous); queue.append(previous)
            assert reachable['layers'][name]['status'] == 'exact'
            assert reachable['layers'][name]['nonmotor_reaching_any_motor'] == ca.ratio(len(seen & nonmotor), len(nonmotor))


@pytest.mark.parametrize('field,value', [
    ('ids', np.array([1, 1, 2, 3, 4, 5])), ('ids', np.array([True] * 6)),
    ('ids', np.arange(6, dtype=float)), ('ids', np.arange(6).reshape(2, 3)),
    ('motor', np.array([-1])), ('motor', np.array([6])), ('motor', np.array([4, 4])),
    ('motor', np.array([True])), ('signs', np.array([1, 1, 2, 1, 1, 1])),
    ('signs', np.array([1., 1., np.nan, 1., 1., 1.])), ('signs', np.ones(6, bool))])
def test_invalid_registry_rejected(field, value):
    ids, motor, signs, edges = fixture_graph()
    args = {'ids': ids, 'motor': motor, 'signs': signs}; args[field] = value
    with pytest.raises(ValueError): ca.audit_arrays(**args, chunks=[edges])


def test_endpoint_validation_and_exact_wide_counts():
    ids, motor, signs, edges = fixture_graph()
    edges['count'] = np.iinfo(np.uint32).max
    assert ca.audit_arrays(ids, motor, signs, [edges])['stored']['stored_synapse_count_sum'] == 8 * (2**32 - 1)
    edges['post'][0] = 100
    with pytest.raises(ValueError, match='endpoint'): ca.audit_arrays(ids, motor, signs, [edges])
    with pytest.raises(ValueError, match='canonical'): ca.audit_arrays(ids, motor, signs, [np.zeros(5)])


@pytest.mark.parametrize('option,value', [
    ('chunk_rows', 0), ('chunk_rows', True), ('chunk_rows', 1_000_001),
    ('min_count', 0), ('min_count', 1.2), ('min_count', True),
    ('max_passes', 0), ('max_passes', 1025), ('max_passes', True),
    ('wall_limit', 0), ('wall_limit', float('nan')), ('wall_limit', float('inf')),
    ('wall_limit', True), ('reachability', 1)])
def test_invalid_budgets_rejected(tmp_path, option, value):
    with pytest.raises(ValueError): ca.audit_graph(tmp_path, **{option: value})


def test_locked_file_audit_is_read_only_and_has_provenance(tmp_path):
    path = tmp_path/'graph'; lock = disk_fixture(path)
    result = ca.audit_graph(path, lock=lock, reachability=True, chunk_rows=1)
    assert result['status'] == 'measured'
    assert result['provenance']['packaged_lock'] is False
    assert result['provenance']['file_sha256'] == lock['files']
    assert result['registered']['neurons'] == 6
    assert result['reachability']['status'] == 'exact'
    assert {n: hashlib.sha256((path/n).read_bytes()).hexdigest() for n in ca.FILES} == lock['files']


@pytest.mark.parametrize('name', sorted(ca.FILES))
def test_every_input_hash_checked(tmp_path, name):
    path = tmp_path/'graph'; lock = disk_fixture(path)
    with (path/name).open('ab') as stream: stream.write(b'x')
    with pytest.raises(ValueError, match='hash mismatch'): ca.audit_graph(path, lock=lock)


def test_inplace_annotation_mutation_during_scan_rejected(tmp_path, monkeypatch):
    path = tmp_path/'graph'; lock = disk_fixture(path); original = ca.audit_arrays
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (path/'neurons.parquet').write_bytes(b'changed')
        return result
    monkeypatch.setattr(ca, 'audit_arrays', mutate)
    with pytest.raises(ValueError, match='changed during'): ca.audit_graph(path, lock=lock)


@pytest.mark.parametrize('field,value', [('neurons', True), ('edges', -1), ('dataset', ''), ('files', {})])
def test_bad_lock_rejected(tmp_path, field, value):
    path = tmp_path/'graph'; lock = disk_fixture(path); lock[field] = value
    with pytest.raises(ValueError): ca.audit_graph(path, lock=lock)


def test_partial_or_wrong_length_records_rejected_even_with_matching_hash(tmp_path):
    path = tmp_path/'graph'; lock = disk_fixture(path)
    (path/'edges.bin').write_bytes((path/'edges.bin').read_bytes() + b'x')
    lock['files']['edges.bin'] = hashlib.sha256((path/'edges.bin').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='byte count'): ca.audit_graph(path, lock=lock)


@pytest.mark.parametrize('version', [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize('dtype', ['<i8', '>u8', 'i1', '<f8'])
def test_numeric_npy_vectors_are_checked_without_pickle(version, dtype):
    values = np.arange(4).astype(dtype); stream = io.BytesIO()
    np.lib.format.write_array(stream, values, version=version)
    assert np.array_equal(ca._read_vector(stream, 4), values)


@pytest.mark.parametrize('change', ['truncated', 'extra', 'huge_shape', 'bool_shape', 'object', 'two_dimensional'])
def test_npy_preflight_rejects_malformed_vectors(change):
    header = {'descr': '<i8', 'fortran_order': False, 'shape': (4,)}
    payload = b'\x00' * 32
    if change == 'truncated': payload = payload[:-1]
    if change == 'extra': payload += b'x'
    if change == 'huge_shape': header['shape'] = (10**12,)
    if change == 'bool_shape': header['shape'] = (True,)
    if change == 'object': header['descr'] = '|O'
    if change == 'two_dimensional': header['shape'] = (2, 2)
    text = (repr(header) + '\n').encode()
    stream = io.BytesIO(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(text)) + text + payload)
    with pytest.raises(ValueError): ca._read_vector(stream, 6)


def test_timeout_is_not_a_negative_biological_result(tmp_path, monkeypatch):
    path = tmp_path/'graph'; lock = disk_fixture(path)
    ticks = iter([0.] + [2.] * 100)
    monkeypatch.setattr(ca.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError): ca.audit_graph(path, lock=lock, wall_limit=1.)


def test_cli_missing_data_writes_failure_and_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root/'fly_effect.py'), 'audit-connectivity', '--graph',
               str(tmp_path/'missing'), '--out', str(tmp_path/'output')]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 2
    receipt = (tmp_path/'output/result.json').read_bytes()
    assert json.loads(receipt)['measured_census'] is None
    assert json.loads(receipt)['walking_claimed'] is False
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert (tmp_path/'output/result.json').read_bytes() == receipt


def test_cli_success_passes_options_and_records_synthetic_provenance(tmp_path, monkeypatch, capsys):
    import fly_effect
    path = tmp_path/'graph'; lock = disk_fixture(path); original = ca.audit_graph
    monkeypatch.setattr(ca, 'audit_graph', lambda directory, **kw: original(directory, lock=lock, **kw))
    monkeypatch.setattr(sys, 'argv', ['fly-effect', 'audit-connectivity', '--graph', str(path),
                                   '--out', str(tmp_path/'result'), '--reachability', '--max-passes', '1', '--min-count', '5'])
    fly_effect.main()
    result = json.loads((tmp_path/'result/result.json').read_text())
    assert result['dataset'] == 'synthetic-test-only'
    assert result['reachability']['status'] == 'lower_bound'
    assert json.loads(capsys.readouterr().out) == result


def test_audit_import_does_not_import_brain_or_body():
    code = "import sys; import organism_core.connectivity_audit; assert not {'brian2','mujoco','flygym'} & set(sys.modules)"
    assert subprocess.run([sys.executable, '-c', code], capture_output=True).returncode == 0
