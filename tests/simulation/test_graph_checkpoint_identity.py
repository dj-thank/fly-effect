"""Data-free graph fixtures check compatibility, not biological performance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

b = pytest.importorskip('brian2')
from organism_core import brain as neural
from organism_core.checkpoint import save, load
from organism_core.config import SOURCE_ROOT


@pytest.fixture
def graph(tmp_path, monkeypatch):
    directory = tmp_path/'graph'
    directory.mkdir()
    # load_graph requires 815 motor identities; all values here are synthetic.
    np.save(directory/'body_ids.npy', np.arange(816, dtype=np.int64))
    np.save(directory/'motor_indices.npy', np.arange(815, dtype=np.int64))
    np.save(directory/'glutamate_inhibitory_hypothesis_signs.npy', np.ones(816, dtype=np.int8))
    np.array([(0, 1, 300, 0), (1, 2, 300, 1)], dtype=neural.EDGE_DTYPE).tofile(directory/'edges.bin')
    (directory/'neurons.parquet').write_bytes(b'synthetic annotation fixture; not biological data')
    lock = {'dataset': 'synthetic-identity-fixture', 'neurons': 816, 'edges': 2, 'files': {}}

    def write_lock():
        lock['files'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in directory.iterdir()}
        (tmp_path/'graph_lock.json').write_text(json.dumps(lock), encoding='utf-8')

    write_lock()
    monkeypatch.setattr(neural, '__file__', str(tmp_path/'brain.py'))
    monkeypatch.setattr(neural, 'GRAPH', directory)
    return directory, lock, write_lock


def test_same_graph_resumes_pending_neural_events_exactly(graph, tmp_path):
    original = neural.Brain(5)
    original.cells.v[0] = -40*b.mV
    original.run(.001)
    receipt = save(tmp_path/'checkpoint.npz', original.state(), original.graph_identity)
    original.run(.019)
    expected = original.observation()

    resumed = neural.Brain(5)
    resumed.restore(load(receipt['path'], resumed.graph_identity, receipt['sha256']))
    resumed.run(.019)
    actual = resumed.observation()
    assert actual['tick'] == expected['tick'] == 200
    assert len(actual['spike_i']) >= 3
    for name in ('v_mV', 'g_mV', 'spike_i', 'spike_t'):
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize('changed', ['ids', 'motor', 'signs', 'annotations', 'dataset'])
def test_same_edges_do_not_allow_other_graph_changes(graph, tmp_path, changed):
    directory, lock, write_lock = graph
    original = neural.Brain(5)
    receipt = save(tmp_path/'checkpoint.npz', original.state(), original.graph_identity)

    if changed == 'ids':
        ids = np.load(directory/'body_ids.npy')
        ids[[0, 1]] = ids[[1, 0]]
        np.save(directory/'body_ids.npy', ids)
    elif changed == 'motor':
        np.save(directory/'motor_indices.npy', np.arange(1, 816, dtype=np.int64))
    elif changed == 'signs':
        signs = np.load(directory/'glutamate_inhibitory_hypothesis_signs.npy')
        signs[0] = -1
        np.save(directory/'glutamate_inhibitory_hypothesis_signs.npy', signs)
    elif changed == 'annotations':
        (directory/'neurons.parquet').write_bytes(b'different synthetic annotation fixture')
    else:
        lock['dataset'] = 'different-synthetic-dataset'
    write_lock()

    current = neural.Brain(5)
    assert original.graph_hash == current.graph_hash
    assert original.graph_manifest_hash != current.graph_manifest_hash
    with pytest.raises(ValueError, match='identity.*graph_manifest_sha256'):
        load(receipt['path'], current.graph_identity, receipt['sha256'])


def test_lock_formatting_does_not_change_graph_identity(graph, tmp_path):
    _, lock, _ = graph
    original = neural.Brain(5)
    reordered = dict(reversed(list(lock.items())))
    reordered['files'] = dict(reversed(list(lock['files'].items())))
    (tmp_path/'graph_lock.json').write_text(json.dumps(reordered, indent=4)+'\n', encoding='utf-8')
    assert neural.Brain(5).graph_identity == original.graph_identity


def test_changed_file_without_updated_lock_fails_integrity(graph):
    directory, _, _ = graph
    (directory/'neurons.parquet').write_bytes(b'unlocked fixture change')
    with pytest.raises(ValueError, match='Graph integrity mismatch: neurons.parquet'):
        neural.Brain(5)


@pytest.mark.parametrize('missing', ['edges.bin', 'body_ids.npy', 'motor_indices.npy',
                                    'glutamate_inhibitory_hypothesis_signs.npy', 'neurons.parquet'])
def test_graph_lock_cannot_omit_consumed_artifacts(graph, tmp_path, missing):
    _, lock, _ = graph
    del lock['files'][missing]
    (tmp_path/'graph_lock.json').write_text(json.dumps(lock), encoding='utf-8')
    with pytest.raises(ValueError, match='Graph lock missing required artifacts: '+missing):
        neural.Brain(5)


@pytest.mark.parametrize('files', [None, []])
def test_graph_lock_files_must_be_a_mapping(graph, tmp_path, files):
    _, lock, _ = graph
    lock['files'] = files
    (tmp_path/'graph_lock.json').write_text(json.dumps(lock), encoding='utf-8')
    with pytest.raises(ValueError, match='Graph lock files must be a dictionary'):
        neural.Brain(5)


def test_legacy_identity_is_not_relabelled(graph, tmp_path):
    current = neural.Brain(5)
    receipt = save(tmp_path/'legacy.npz', current.state(), {'graph_sha256': current.graph_hash})
    before = Path(receipt['path']).read_bytes()
    with pytest.raises(ValueError, match='identity.*graph_manifest_sha256'):
        load(receipt['path'], current.graph_identity, receipt['sha256'])
    assert Path(receipt['path']).read_bytes() == before


def test_engine_rejects_graph_mismatch_before_mutating_runtime(graph, tmp_path, monkeypatch):
    directory, _, write_lock = graph
    original = neural.Brain(5)
    receipt = save(tmp_path/'checkpoint.npz', {'neural': original.state()}, original.graph_identity)
    signs = np.load(directory/'glutamate_inhibitory_hypothesis_signs.npy')
    signs[0] = -1
    np.save(directory/'glutamate_inhibitory_hypothesis_signs.npy', signs)
    write_lock()
    current = neural.Brain(5)
    before = current.observation()
    weights = np.asarray(current.syn.w[:]).copy()
    restored = []
    monkeypatch.setattr(current, 'restore', lambda state: restored.append('neural'))

    # Engine.restore is exercised with a Body stub; MuJoCo/assets are not needed
    # to establish that identity rejection precedes every subsystem mutation.
    body_module = types.ModuleType('organism_core.body')
    body_module.Body = object
    monkeypatch.setitem(sys.modules, 'organism_core.body', body_module)
    spec = importlib.util.spec_from_file_location('organism_core._identity_test_engine', SOURCE_ROOT/'organism_core/engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = module.Engine.__new__(module.Engine)
    engine.identity = current.graph_identity
    engine.brain = current
    engine.body = types.SimpleNamespace(restore=lambda state: restored.append('body'))
    with pytest.raises(ValueError, match='identity.*graph_manifest_sha256'):
        engine.restore(receipt['path'], receipt['sha256'])
    assert restored == []
    np.testing.assert_array_equal(np.asarray(current.syn.w[:]), weights)
    after = current.observation()
    assert before['tick'] == after['tick']
    for name in ('v_mV', 'g_mV', 'spike_i', 'spike_t'):
        np.testing.assert_array_equal(after[name], before[name])
