"""Bounded fixtures exercise allocation guards; no large array is allocated."""
import io
import json
import struct
import zipfile
import numpy as np
import pytest
from organism_core import checkpoint as codec


def write_archive(path, payload, *, metadata=None):
    meta = io.BytesIO()
    np.save(meta, np.array(metadata or json.dumps(
        {'schema': 1, 'identity': {}, 'tree': {'array': 'a0'}})))
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('metadata.npy', meta.getvalue())
        archive.writestr('a0.npy', payload)
    return path


def npy_header(shape, descr='<f8', version=(1, 0), payload=b''):
    header = repr({'shape': shape, 'fortran_order': False, 'descr': descr}) + '\n'
    raw = header.encode('utf-8' if version == (3, 0) else 'latin1')
    width = '<H' if version == (1, 0) else '<I'
    return b'\x93NUMPY' + bytes(version) + struct.pack(width, len(raw)) + raw + payload


@pytest.mark.parametrize('shape,payload', [((10**15,), b''), ((2,), b'12345678'),
    ((1,), b'x'*16), ((2**63, 2**63), b''), ((-1,), b''), ((True,), b'')])
def test_bad_shape_or_payload_is_rejected_before_numpy_load(tmp_path, monkeypatch, shape, payload):
    path = write_archive(tmp_path/'bad.npz', npy_header(shape, payload=payload))
    def forbidden(*args, **kwargs):
        raise AssertionError('np.load must not receive a forged allocation')
    monkeypatch.setattr(codec.np, 'load', forbidden)
    with pytest.raises(ValueError):
        codec.load(path, {})


@pytest.mark.parametrize('payload', [b'\x93NUMPY\x01\x00\x01',
    b'\x93NUMPY\x01\x00\xff\xff', b'\x93NUMPY\x01\x00\x08\x00x',
    npy_header((1,), '|O', payload=b'12345678'), npy_header((1,), version=(4, 0))])
def test_invalid_header_rejected(tmp_path, payload):
    with pytest.raises(ValueError):
        codec.load(write_archive(tmp_path/'bad.npz', payload), {})


@pytest.mark.parametrize('version', [(1, 0), (2, 0), (3, 0)])
def test_supported_npy_versions(tmp_path, version):
    array = np.arange(6, dtype='>i4').reshape(2, 3)
    stream = io.BytesIO(); np.lib.format.write_array(stream, array, version=version)
    actual = codec.load(write_archive(tmp_path/'ok.npz', stream.getvalue()), {})
    np.testing.assert_array_equal(actual, array)
    assert actual.dtype == array.dtype


def test_unicode_structured_npy_v3(tmp_path):
    array = np.zeros(3, dtype=[('角度', '<f8'), ('joint', '<i4')])
    stream = io.BytesIO(); np.lib.format.write_array(stream, array, version=(3, 0))
    actual = codec.load(write_archive(tmp_path/'ok.npz', stream.getvalue()), {})
    np.testing.assert_array_equal(actual, array)
    assert actual.dtype == array.dtype


@pytest.mark.parametrize('value', ['1e999', '-1e999', 'NaN', 'Infinity', '-Infinity'])
def test_json_overflow_never_becomes_checkpoint_state(tmp_path, value):
    path = tmp_path/'bad.npz'
    np.savez(path, metadata=np.array('{"schema":1,"identity":{},"tree":{"scalar":' + value + '}}'))
    with pytest.raises(ValueError, match='Nonfinite'):
        codec.load(path, {})


@pytest.mark.parametrize('name', ['max_uncompressed_bytes', 'max_metadata_bytes'])
@pytest.mark.parametrize('limit', [0, -1, True, 1.5, float('inf'), '1000'])
def test_invalid_resource_limits(tmp_path, name, limit):
    with pytest.raises(ValueError, match='positive integer'):
        codec.load(tmp_path/'unused', {}, **{name: limit})


def test_byte_budgets_include_headers_and_metadata(tmp_path, monkeypatch):
    path = tmp_path/'state.npz'; receipt = codec.save(path, {'x': np.zeros(100)}, {})
    with zipfile.ZipFile(path) as archive:
        total = sum(info.file_size for info in archive.infolist())
        meta = archive.getinfo('metadata.npy').file_size
    codec.load(path, {}, receipt['sha256'], max_uncompressed_bytes=total, max_metadata_bytes=meta)
    for kwargs in ({'max_uncompressed_bytes': total-1}, {'max_metadata_bytes': meta-1}):
        with pytest.raises(ValueError, match='byte budget'):
            codec.load(path, {}, **kwargs)


@pytest.mark.parametrize('array', [np.arange(5), np.zeros((0, 3)), np.array(3.25),
    np.asfortranarray(np.arange(12).reshape(3, 4)), np.array(['日本語', 'fly']),
    np.array(['2026-09-11'], dtype='datetime64[D]')])
def test_restored_arrays_survive_close_and_do_not_alias_disk(tmp_path, array):
    path = tmp_path/'state.npz'; receipt = codec.save(path, {'x': array}, {})
    first = codec.load(path, {}, receipt['sha256'])['x']
    second = codec.load(path, {}, receipt['sha256'])['x']
    np.testing.assert_array_equal(first, array)
    assert first.flags.writeable
    assert not np.shares_memory(first, second)
    if first.size:
        first.reshape(-1)[0] = first.reshape(-1)[-1]
    np.testing.assert_array_equal(codec.load(path, {}, receipt['sha256'])['x'], array)
