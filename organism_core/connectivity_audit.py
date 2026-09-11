"""Read-only graph census. Registry, topology and nonzero model weights are distinct.

No Brian2/MuJoCo import, dense adjacency, simulation, or biological acceptance.
The file entry point only accepts bytes matching the packaged graph lock.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Callable, Iterable

import numpy as np

EDGE_DTYPE = np.dtype([('pre', '<u4'), ('post', '<u4'), ('count', '<u4'), ('source_row', '<u8')])
SIGN_FILE = 'glutamate_inhibitory_hypothesis_signs.npy'
FILES = {'edges.bin', 'body_ids.npy', 'motor_indices.npy', SIGN_FILE, 'neurons.parquet'}
LAYERS = ('structural', 'nonzero_model_weight')
BLOCKS = ('nonmotor_to_nonmotor', 'nonmotor_to_motor', 'motor_to_nonmotor', 'motor_to_motor')


def ratio(numerator: int, denominator: int) -> dict:
    """Never turn an empty denominator into a fabricated zero percent."""
    return {'numerator': int(numerator), 'denominator': int(denominator),
            'percent': 100.0 * numerator / denominator if denominator else None}


def _integer(value, name: str, minimum: int = 1, maximum: int = 2**32 - 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not minimum <= int(value) <= maximum:
        raise ValueError(f'{name} outside supported range')
    return int(value)


def _registry(ids, motor, signs):
    ids, motor, signs = np.asarray(ids), np.asarray(motor), np.asarray(signs)
    for name, values in (('body IDs', ids), ('motor indices', motor)):
        if values.ndim != 1 or values.dtype.kind not in 'iu':
            raise ValueError(f'{name} must be a one-dimensional integer array')
        if np.any(values < 0) or len(np.unique(values)) != len(values):
            raise ValueError(f'{name} must be nonnegative and unique')
    n = len(ids)
    if len(motor) and int(motor.max()) >= n:
        raise ValueError('Motor index outside registered neurons')
    if (signs.shape != (n,) or signs.dtype.kind not in 'iuf'
            or not np.isin(signs, (-1, 0, 1)).all()):
        raise ValueError('Signs must be a numeric -1/0/1 vector aligned with IDs')
    mask = np.zeros(n, dtype=bool)
    mask[motor.astype(np.intp)] = True
    return ids, mask, signs


def _chunk(chunk, n: int):
    if not isinstance(chunk, np.ndarray) or chunk.ndim != 1 or chunk.dtype != EDGE_DTYPE:
        raise ValueError('Expected one-dimensional canonical edge records')
    if len(chunk) and (int(chunk['pre'].max()) >= n or int(chunk['post'].max()) >= n):
        raise ValueError('Edge endpoint outside registered neurons')
    return chunk['pre'], chunk['post'], chunk['count']


def _new_layer(n: int):
    return {'incoming': np.zeros(n, bool), 'outgoing': np.zeros(n, bool),
            'nonself': np.zeros(n, bool), 'direct_motor': np.zeros(n, bool),
            'rows': [0] * 4, 'counts': [0] * 4}


def _finish_layer(layer, motor):
    incident = layer['incoming'] | layer['outgoing']
    blocks = {key: {'edge_rows': layer['rows'][i], 'stored_synapse_count_sum': layer['counts'][i]}
              for i, key in enumerate(BLOCKS)}
    result = {'edge_rows': sum(layer['rows']), 'stored_synapse_count_sum': sum(layer['counts']),
              'blocks': blocks, 'groups': {}}
    for name, population in (('all', np.ones(len(motor), bool)), ('motor', motor), ('nonmotor', ~motor)):
        size = int(population.sum())
        result['groups'][name] = {
            'registered': size,
            'incident': ratio(int((population & incident).sum()), size),
            'connected_to_other_neurons': ratio(int((population & layer['nonself']).sum()), size),
            'with_input': ratio(int((population & layer['incoming']).sum()), size),
            'with_output': ratio(int((population & layer['outgoing']).sum()), size),
            'isolated': ratio(int((population & ~incident).sum()), size),
            'directly_targets_motor': ratio(int((population & layer['direct_motor']).sum()), size)}
    return result


def audit_arrays(ids, motor, signs, chunks: Iterable[np.ndarray], *, min_count=1,
                 check: Callable[[], None] = lambda: None) -> dict:
    """Census every supplied row; caller labels provenance, never assumes biology.

Duplicate endpoint pairs remain separate rows. Thresholds apply to each stored
row, not to an unperformed aggregation of duplicate pairs.
"""
    min_count = _integer(min_count, 'min_count')
    ids, motor, signs = _registry(ids, motor, signs)
    n = len(ids)
    layers = {name: _new_layer(n) for name in LAYERS}
    rows = count_sum = zero = self_rows = 0
    for chunk in chunks:
        check()
        pre, post, counts = _chunk(chunk, n)
        rows += len(chunk)
        # uint32 counts and bounded file chunks make each sum exact in uint64.
        count_sum += int(counts.sum(dtype=np.uint64))
        zero += int(np.count_nonzero(counts == 0))
        self_rows += int(np.count_nonzero(pre == post))
        selected = counts >= min_count
        for name in LAYERS:
            take = selected if name == 'structural' else selected & (signs[pre] != 0)
            a, b, c = pre[take], post[take], counts[take]
            layer = layers[name]
            layer['outgoing'][a] = True
            layer['incoming'][b] = True
            other = a != b
            layer['nonself'][a[other]] = True
            layer['nonself'][b[other]] = True
            layer['direct_motor'][a[motor[b]]] = True
            code = motor[a].astype(np.uint8) * 2 + motor[b].astype(np.uint8)
            for i in range(4):
                subset = code == i
                layer['rows'][i] += int(np.count_nonzero(subset))
                layer['counts'][i] += int(c[subset].sum(dtype=np.uint64))
        check()
    m = int(motor.sum())
    return {'schema': 1, 'kind': 'graph_census_not_biological_validation',
            'registered': {'neurons': n, 'motor': ratio(m, n), 'nonmotor': ratio(n - m, n),
                           'sign_zero': ratio(int(np.count_nonzero(signs == 0)), n)},
            'stored': {'edge_rows': rows, 'stored_synapse_count_sum': count_sum,
                       'zero_count_rows': zero, 'self_edge_rows': self_rows,
                       'unique_endpoint_pairs': None},
            'selection': {'minimum_stored_row_count': min_count, 'pair_aggregation': False},
            'layers': {name: _finish_layer(layer, motor) for name, layer in layers.items()},
            'reachability': {'status': 'not_measured'},
            'biological_validation': False, 'walking_claimed': False,
            'limitations': ['Nonmotor means outside the supplied motor registry, not unrelated to movement.',
                           'Nonzero model weight is not observed activity or proof of transmission.',
                           'Edge rows and stored count sums are not independently reconstructed EM contacts.',
                           'Structural incidence does not imply a single connected component.',
                           'No measured firing, muscle correspondence, learning, or behavior is inferred.']}


def motor_reachability(ids, motor, signs, chunks: Callable[[], Iterable[np.ndarray]], *,
                       min_count=1, max_passes=64, check: Callable[[], None] = lambda: None) -> dict:
    """Bounded reverse traversal using O(V + chunk) memory, not an E-sized CSR.

Each complete pass uses a frozen frontier, so edge order does not change the
bounded result. An exhausted pass budget is a lower bound, never exact coverage.
Inhibitory edges count as possible influence, not guaranteed motor activation.
"""
    min_count = _integer(min_count, 'min_count')
    max_passes = _integer(max_passes, 'max_passes', maximum=1024)
    ids, motor_mask, signs = _registry(ids, motor, signs)
    reach = {name: motor_mask.copy() for name in LAYERS}
    done = {name: not motor_mask.any() or motor_mask.all() for name in LAYERS}
    passes = 0
    while not all(done.values()) and passes < max_passes:
        following = {name: reached.copy() for name, reached in reach.items()}
        for chunk in chunks():
            check()
            pre, post, counts = _chunk(chunk, len(ids))
            for name in LAYERS:
                if done[name]:
                    continue
                take = (counts >= min_count) & reach[name][post]
                if name == 'nonzero_model_weight':
                    take &= signs[pre] != 0
                following[name][pre[take]] = True
            check()
        passes += 1
        for name in LAYERS:
            done[name] = bool(np.array_equal(reach[name], following[name]) or following[name].all())
        reach = following
    results = {}
    for name in LAYERS:
        results[name] = {'status': 'exact' if done[name] else 'lower_bound',
                         'nonmotor_reaching_any_motor': ratio(int((reach[name] & ~motor_mask).sum()),
                                                             int((~motor_mask).sum()))}
    return {'status': 'exact' if all(done.values()) else 'lower_bound', 'completed_passes': passes,
            'maximum_passes': max_passes, 'target_count': int(motor_mask.sum()),
            'direction': 'neuron_to_any_motor', 'layers': results,
            'note': 'Path existence only. Inhibitory edges are included; no activation or behavior claim.'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _read_vector(stream, max_length: int) -> np.ndarray:
    """Validate a bounded scalar-numeric NPY before allocating its payload."""
    stream.seek(0)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError('Unsupported NPY version')
    width = 2 if version == (1, 0) else 4
    length_bytes = stream.read(width)
    if len(length_bytes) != width:
        raise ValueError('Truncated NPY length')
    length = struct.unpack('<H' if width == 2 else '<I', length_bytes)[0]
    if not 0 < length <= 4096:
        raise ValueError('NPY header exceeds audit budget')
    header = stream.read(length)
    if len(header) != length:
        raise ValueError('Truncated NPY header')
    try:
        value = ast.literal_eval(header.decode('utf-8' if version == (3, 0) else 'latin1'))
    except (ValueError, SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError('Invalid NPY header') from error
    if not isinstance(value, dict) or set(value) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('Unexpected NPY header fields')
    shape = value['shape']
    if (not isinstance(shape, tuple) or len(shape) != 1 or type(shape[0]) is not int
            or not 0 <= shape[0] <= max_length or type(value['fortran_order']) is not bool
            or not isinstance(value['descr'], str)):
        raise ValueError('NPY must be a bounded one-dimensional numeric vector')
    dtype = np.dtype(value['descr'])
    if dtype.kind not in 'iuf' or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError('Unsupported audit vector dtype')
    size = shape[0] * dtype.itemsize
    position = stream.tell()
    stream.seek(0, 2)
    if stream.tell() - position != size or size > 64 * 1024**2:
        raise ValueError('NPY payload size mismatch or budget exceeded')
    stream.seek(position)
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError('Truncated NPY payload')
    return np.frombuffer(payload, dtype=dtype)


def audit_graph(directory: Path, *, chunk_rows=250_000, wall_limit=120.0,
                min_count=1, reachability=False, max_passes=64, lock=None) -> dict:
    """Audit locked local files. Custom locks exist for programmatic test fixtures.

A wall limit is checked between operations; it is not an OS-level I/O deadline
or memory sandbox. Original files and the packaged lock are never written.
"""
    chunk_rows = _integer(chunk_rows, 'chunk_rows', maximum=1_000_000)
    min_count = _integer(min_count, 'min_count')
    max_passes = _integer(max_passes, 'max_passes', maximum=1024)
    if (isinstance(wall_limit, bool) or not isinstance(wall_limit, (int, float))
            or not math.isfinite(wall_limit) or not 0 < wall_limit <= 3600):
        raise ValueError('wall_limit must be finite and in (0, 3600] seconds')
    if type(reachability) is not bool:
        raise ValueError('reachability must be boolean')
    started = time.monotonic()
    def check():
        if time.monotonic() - started > wall_limit:
            raise TimeoutError('Graph audit wall budget exhausted; no complete census certified')
    packaged = lock is None
    if packaged:
        lock = json.loads(Path(__file__).with_name('graph_lock.json').read_text(encoding='utf-8'),
                          object_pairs_hook=_unique_object)
    if not isinstance(lock, dict) or not isinstance(lock.get('dataset'), str) or not lock['dataset']:
        raise ValueError('Dataset identity required')
    n = _integer(lock.get('neurons'), 'locked neurons', minimum=0)
    edge_count = _integer(lock.get('edges'), 'locked edges', minimum=0, maximum=2**63 - 1)
    if not isinstance(lock.get('files'), dict) or set(lock['files']) != FILES:
        raise ValueError('Exact five locked graph files required')
    for digest in lock['files'].values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid locked SHA256')
    def digest_file(stream):
        stream.seek(0)
        h = hashlib.sha256()
        while True:
            check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
        return h.hexdigest()
    with ExitStack() as stack:
        streams = {name: stack.enter_context((Path(directory) / name).open('rb')) for name in sorted(FILES)}
        for name, stream in streams.items():
            if digest_file(stream) != lock['files'][name]:
                raise ValueError('Graph hash mismatch: ' + name)
        ids = _read_vector(streams['body_ids.npy'], n)
        motor = _read_vector(streams['motor_indices.npy'], n)
        signs = _read_vector(streams[SIGN_FILE], n)
        if len(ids) != n:
            raise ValueError('Locked neuron count mismatch')
        edges = streams['edges.bin']
        edges.seek(0, 2)
        if edges.tell() != edge_count * EDGE_DTYPE.itemsize:
            raise ValueError('Locked edge byte count mismatch')
        def chunks():
            edges.seek(0)
            while True:
                check()
                payload = edges.read(chunk_rows * EDGE_DTYPE.itemsize)
                if not payload:
                    break
                if len(payload) % EDGE_DTYPE.itemsize:
                    raise ValueError('Truncated edge record')
                yield np.frombuffer(payload, dtype=EDGE_DTYPE)
        result = audit_arrays(ids, motor, signs, chunks(), min_count=min_count, check=check)
        if result['stored']['edge_rows'] != edge_count:
            raise ValueError('Edge file changed during audit')
        if reachability:
            result['reachability'] = motor_reachability(ids, motor, signs, chunks,
                min_count=min_count, max_passes=max_passes, check=check)
        # Rehash the same descriptors, detecting ordinary in-place changes.
        for name, stream in streams.items():
            if digest_file(stream) != lock['files'][name]:
                raise ValueError('Graph changed during audit: ' + name)
    result.update(status='measured', dataset=lock['dataset'],
                  provenance={'packaged_lock': packaged, 'file_sha256': dict(lock['files']),
                              'lock_sha256': hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()},
                  budget={'chunk_rows': chunk_rows, 'wall_limit_s': wall_limit,
                          'elapsed_s': time.monotonic() - started},
                  annotation_breakdown={'status': 'not_measured', 'reason': 'No cell-type classification inferred from motor membership'})
    return result
