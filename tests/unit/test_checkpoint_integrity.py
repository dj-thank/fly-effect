"""Malformed inputs are negative fixtures, never biological evidence."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from organism_core import checkpoint as codec


def archive(path, tree, **arrays):
    np.savez(path, metadata=np.array(json.dumps({'schema': 1, 'identity': {}, 'tree': tree})), **arrays)
    return path


@pytest.mark.parametrize('tree', [
    None, [], {}, {'list': None}, {'tuple': {}}, {'dict': {}},
    {'scalar': []}, {'scalar': {}}, {'array': 'metadata'}, {'array': 'absent'},
    {'array': 0}, {'array': '../a0'}, {'array': 'a01'}, {'unknown': 1},
    {'scalar': 1, 'array': 'a0'}, {'dict': [['x']]}, {'dict': [[1, {'scalar': 1}]]},
    {'dict': [['x', {'scalar': 1}], ['x', {'scalar': 2}]]},
    {'list': [{'array': 'a0'}, {'array': 'a0'}]},
])
def test_rejects_malformed_tree(tmp_path, tree):
    with pytest.raises(ValueError):
        codec.load(archive(tmp_path/'bad.npz', tree, a0=np.zeros(1)), {})


def test_rejects_orphan_array(tmp_path):
    with pytest.raises(ValueError, match='Unreferenced'):
        codec.load(archive(tmp_path/'bad.npz', {'scalar': 1}, a0=np.zeros(1)), {})


@pytest.mark.parametrize('metadata', [
    '[]', 'null', '{"schema":true,"identity":{},"tree":{"scalar":1}}',
    '{"schema":1,"identity":null,"tree":{"scalar":1}}',
    '{"schema":1,"schema":1,"identity":{},"tree":{"scalar":1}}',
    '{"schema":1,"identity":{},"tree":{"scalar":NaN}}',
])
def test_metadata_is_strict(tmp_path, metadata):
    path = tmp_path/'bad.npz'; np.savez(path, metadata=np.array(metadata))
    with pytest.raises(ValueError):
        codec.load(path, {})


def test_metadata_shape_is_strict(tmp_path):
    path = tmp_path/'bad.npz'; np.savez(path, metadata=np.array(['{}']))
    with pytest.raises(ValueError, match='Unicode scalar'):
        codec.load(path, {})


def test_duplicate_zip_member_rejected(tmp_path):
    path = archive(tmp_path/'bad.npz', {'scalar': 1})
    with zipfile.ZipFile(path, 'a') as z:
        with pytest.warns(UserWarning):
            z.writestr('metadata.npy', b'not-a-valid-array')
    with pytest.raises(ValueError, match='duplicate'):
        codec.load(path, {})


@pytest.mark.parametrize('stage', ['write', 'fsync', 'replace'])
def test_failed_save_keeps_previous_checkpoint_and_cleans_temp(tmp_path, monkeypatch, stage):
    path = tmp_path/'state.npz'; old = codec.save(path, {'tick': 1}, {})
    def fail(*args, **kwargs):
        if stage == 'write':
            args[0].write(b'partial')
        raise OSError('injected failure')
    if stage == 'write': monkeypatch.setattr(codec.np, 'savez_compressed', fail)
    elif stage == 'fsync': monkeypatch.setattr(codec.os, 'fsync', fail)
    else: monkeypatch.setattr(codec.os, 'replace', fail)
    with pytest.raises(OSError, match='injected'):
        codec.save(path, {'tick': 2}, {})
    assert codec.load(path, {}, old['sha256']) == {'tick': 1}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(__import__('sys').platform == 'win32', reason='POSIX open-file replacement semantics')
def test_hash_and_decode_use_same_open_file(tmp_path, monkeypatch):
    path = tmp_path/'state.npz'; old = codec.save(path, {'tick': 1}, {})
    replacement = tmp_path/'replacement.npz'; codec.save(replacement, {'tick': 2}, {})
    real = hashlib.file_digest
    def replace_after_hash(stream, algorithm):
        digest = real(stream, algorithm)
        replacement.replace(path)
        return digest
    monkeypatch.setattr(codec.hashlib, 'file_digest', replace_after_hash)
    assert codec.load(path, {}, old['sha256']) == {'tick': 1}
    monkeypatch.setattr(codec.hashlib, 'file_digest', real)
    assert codec.load(path, {}) == {'tick': 2}


@pytest.mark.parametrize('value', ['', 'bad', 0, True, 'A'*64])
def test_invalid_expected_hash_is_not_silently_disabled(tmp_path, value):
    path = archive(tmp_path/'state.npz', {'scalar': 1})
    with pytest.raises(ValueError, match='SHA256'):
        codec.load(path, {}, value)


def test_all_supported_types_roundtrip_without_identity_rewrite(tmp_path):
    state = {'none': None, 'scalar': [True, 7, 2.5, '日本語'],
             'tuple': (np.array([1, 2], dtype='>i4'), np.float64(4)),
             'structured': np.array([(2, .5)], dtype=[('i', 'i4'), ('f', 'f8')])}
    identity = {'model': 'original', 'settings': {'seed': 3}}
    path = tmp_path/'state.npz'; receipt = codec.save(path, state, identity)
    assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    actual = codec.load(path, identity, receipt['sha256'])
    assert actual['scalar'] == state['scalar'] and isinstance(actual['tuple'], tuple)
    np.testing.assert_array_equal(actual['tuple'][0], state['tuple'][0])
    np.testing.assert_array_equal(actual['structured'], state['structured'])
