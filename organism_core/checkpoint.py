"""Strict schema-1 checkpoints, without pickle or executable deserialization.

Writes replace the destination only after a complete, flushed archive exists.
Readers hash and decode the same open file, even if its pathname is replaced.
Existing valid schema-1 checkpoints remain readable; no identity is rewritten.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import uuid
import zipfile
import numpy as np


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate checkpoint JSON key: '+key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite checkpoint JSON constant: '+value)


def save(path, state, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('Object arrays forbidden')
            name = f'a{len(arrays)}'
            arrays[name] = value
            return {'array': name}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError('String keys required')
            return {'dict': [[k, encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {'tuple' if isinstance(value, tuple) else 'list': [encode(v) for v in value]}
        if value is None or isinstance(value, (bool, int, float, str)):
            return {'scalar': value}
        raise TypeError('Unsupported checkpoint type: '+str(type(value)))

    if not isinstance(identity, dict):
        raise TypeError('Checkpoint identity must be a dictionary')
    tree = encode(state)
    arrays['metadata'] = np.array(json.dumps(
        {'schema': 1, 'identity': identity, 'tree': tree}, allow_nan=False))
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x+b') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return {'path': str(path), 'sha256': digest}


def load(path, identity, expected_sha256=None):
    if not isinstance(identity, dict):
        raise TypeError('Checkpoint identity must be a dictionary')
    if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256)):
        raise ValueError('Expected checkpoint hash must be a lowercase SHA256')
    # One descriptor closes the pathname replacement gap between verification and decode.
    with Path(path).open('rb') as stream:
        if expected_sha256 is not None:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_sha256:
                raise ValueError('Checkpoint hash mismatch')
            stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            names = archive.namelist()
            if (len(names) != len(set(names)) or 'metadata.npy' not in names
                    or any(n != 'metadata.npy' and not re.fullmatch(r'a(?:0|[1-9][0-9]*)\.npy', n)
                           for n in names)):
                raise ValueError('Invalid or duplicate checkpoint archive members')
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            raw = data['metadata']
            if raw.shape != () or raw.dtype.kind != 'U':
                raise ValueError('Checkpoint metadata must be a Unicode scalar')
            meta = json.loads(str(raw), object_pairs_hook=_unique_object,
                              parse_constant=_invalid_constant)
            if (not isinstance(meta, dict) or set(meta) != {'schema', 'identity', 'tree'}
                    or type(meta['schema']) is not int or meta['schema'] != 1
                    or not isinstance(meta['identity'], dict)):
                raise ValueError('Invalid checkpoint schema')
            if meta['identity'] != identity:
                differences = [k for k in set(meta['identity']) | set(identity)
                               if meta['identity'].get(k) != identity.get(k)]
                raise ValueError('Checkpoint identity mismatch: '+','.join(sorted(differences)))
            used = set()

            def decode(node):
                if not isinstance(node, dict) or len(node) != 1:
                    raise ValueError('Invalid checkpoint node')
                tag, value = next(iter(node.items()))
                if tag == 'array':
                    if (not isinstance(value, str) or not re.fullmatch(r'a(?:0|[1-9][0-9]*)', value)
                            or value not in data.files or value in used):
                        raise ValueError('Invalid or repeated checkpoint array reference')
                    used.add(value)
                    return data[value].copy()
                if tag == 'scalar':
                    if value is not None and type(value) not in (bool, int, float, str):
                        raise ValueError('Invalid checkpoint scalar')
                    return value
                if tag in ('tuple', 'list'):
                    if not isinstance(value, list):
                        raise ValueError('Invalid checkpoint sequence')
                    items = [decode(v) for v in value]
                    return tuple(items) if tag == 'tuple' else items
                if tag == 'dict':
                    if not isinstance(value, list):
                        raise ValueError('Invalid checkpoint dictionary')
                    result = {}
                    for pair in value:
                        if (not isinstance(pair, list) or len(pair) != 2
                                or not isinstance(pair[0], str) or pair[0] in result):
                            raise ValueError('Invalid or duplicate checkpoint dictionary key')
                        result[pair[0]] = decode(pair[1])
                    return result
                raise ValueError('Unsupported checkpoint node')

            state = decode(meta['tree'])
            if used != set(data.files)-{'metadata'}:
                raise ValueError('Unreferenced checkpoint arrays')
            return state
