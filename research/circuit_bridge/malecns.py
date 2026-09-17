"""Reproducible public MaleCNS acquisition and held-out bridge experiment.

Run: python -m research.circuit_bridge.malecns --data /tmp/malecns --out /tmp/result.json
Raw data stays outside git. HHMI Janelia MaleCNS v1.0, CC-BY 4.0:
https://male-cns.janelia.org/download/
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import urllib.request
import numpy as np
from scipy import sparse
from .connectome import (RateConfig, SparseCircuit, array_hash, comparison,
                        pod_basis, simulate)

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
SOURCES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'nt': 'body-neurotransmitters-male-cns-v1.0.feather',
    'weights': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
}
WEIGHT_SHA = 'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1'
EDGE_DTYPE = np.dtype([('pre', '<u4'), ('post', '<u4'), ('count', '<u4'), ('source_row', '<u8')])


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(p)


def acquire(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    manifest = {'source': 'HHMI Janelia MaleCNS v1.0', 'license': 'CC-BY-4.0',
                'source_page': 'https://male-cns.janelia.org/download/', 'files': {}}
    for key, name in SOURCES.items():
        path = directory / name
        url = BASE + name
        h = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=90) as response, path.with_suffix('.part').open('wb') as out:
            generation = response.headers.get('x-goog-generation')
            if not generation or not generation.isdigit():
                raise ValueError('Missing immutable GCS generation')
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                size += len(chunk)
                if size > 1_200_000_000:
                    raise ValueError('Source exceeds public download budget')
                out.write(chunk); h.update(chunk)
        if key == 'weights' and h.hexdigest() != WEIGHT_SHA:
            raise ValueError('Pinned MaleCNS connection source hash changed')
        path.with_suffix('.part').replace(path)
        manifest['files'][key] = {'url': url + '?generation=' + generation,
            'generation': generation, 'sha256': h.hexdigest(), 'bytes': size}
        write_json(directory / 'source-manifest.json', manifest)
    return manifest


def column(frame, choices):
    for key in choices:
        if key in frame.columns:
            return key
    raise ValueError(f'Required column missing: {choices}; available={list(frame.columns)}')


def table_frame(path):
    import pyarrow.feather as feather
    frame = feather.read_table(path).to_pandas()
    if frame.index.name is not None and frame.index.name not in frame.columns:
        frame = frame.reset_index()
    return frame


def import_graph(directory):
    import pyarrow as pa
    d = Path(directory)
    ann = table_frame(d / SOURCES['annotations'])
    nt = table_frame(d / SOURCES['nt'])
    aid = column(ann, ('body', 'bodyId', 'body_id'))
    nid = column(nt, ('body', 'bodyId', 'body_id'))
    ntcol = column(nt, ('consensus_nt', 'predicted_nt', 'nt'))
    diagnostic = {'annotation_columns': list(ann.columns), 'nt_columns': list(nt.columns),
        'annotation_rows': len(ann), 'nt_rows': len(nt)}
    for key in ('status', 'superclass', 'class'):
        if key in ann:
            diagnostic[key] = {str(k): int(v) for k, v in ann[key].value_counts(dropna=False).items()}
    write_json(d / 'schema.json', diagnostic)
    if ann[aid].duplicated().any() or nt[nid].duplicated().any():
        raise ValueError('Ambiguous source ID annotation')
    ids = np.sort(ann[aid].to_numpy(dtype=np.int64))
    if len(ids) != 166700:
        raise ValueError(f'Annotated population differs from upstream 166700: {len(ids)}; see schema.json')
    ann = ann.set_index(aid).loc[ids]
    labels = nt.set_index(nid).reindex(ids)[ntcol].fillna('unknown').astype(str).str.lower()
    signs = labels.map({'acetylcholine': 1, 'gaba': -1, 'glutamate': -1}).fillna(0).to_numpy(dtype=np.int8)
    # Same explicit glutamate-inhibitory hypothesis; NOT receptor-aware physiology.
    motor = np.flatnonzero(ann['superclass'].astype(str).eq('vnc_motor').to_numpy())
    if len(motor) != 815:
        raise ValueError(f'Expected upstream 815 motor annotations, got {len(motor)}')
    stim = np.flatnonzero(ann['type'].astype(str).eq('DNge104').to_numpy())
    if set(map(int, ids[stim])) != {12781, 556329}:
        raise ValueError('Published DNge104 source identities changed')
    path = d / 'edges.bin'
    total = retained = synapses = 0
    with pa.memory_map(str(d / SOURCES['weights']), 'r') as source, path.open('wb') as output:
        reader = pa.ipc.open_file(source)
        names = reader.schema.names
        dummy = type('Columns', (), {'columns': names})()
        pc = column(dummy, ('body_pre', 'bodyId_pre', 'pre'))
        qc = column(dummy, ('body_post', 'bodyId_post', 'post'))
        wc = column(dummy, ('weight', 'syn_count', 'count'))
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre = batch.column(names.index(pc)).to_numpy()
            post = batch.column(names.index(qc)).to_numpy()
            count = batch.column(names.index(wc)).to_numpy()
            p, q = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            keep = ((p < len(ids)) & (q < len(ids)) &
                    (ids[np.minimum(p, len(ids)-1)] == pre) &
                    (ids[np.minimum(q, len(ids)-1)] == post))
            if (count.dtype.kind not in 'iu' or np.any(count <= 0)
                    or np.any(count > np.iinfo(np.uint32).max)):
                raise ValueError('Invalid anatomical synapse counts')
            part = np.empty(int(keep.sum()), dtype=EDGE_DTYPE)
            part['pre'], part['post'], part['count'] = p[keep], q[keep], count[keep]
            part['source_row'] = total + np.flatnonzero(keep)
            part.tofile(output)
            total += len(pre); retained += len(part)
            synapses += int(part['count'].sum(dtype=np.uint64))
    if retained != 25582938:
        raise ValueError(f'Annotated graph differs from upstream 25582938: {retained}')
    edges = np.memmap(path, dtype=EDGE_DTYPE, mode='r')
    np.save(d / 'body_ids.npy', ids, allow_pickle=False)
    report = {'neurons': len(ids), 'connection_rows': retained, 'synapse_sum': synapses,
        'raw_connection_rows': total, 'unannotated_endpoint_rows_excluded': total-retained,
        'motor_neurons': len(motor), 'stimulus_body_ids': ids[stim].tolist(),
        'body_ids_npy_sha256': hashlib.sha256((d/'body_ids.npy').read_bytes()).hexdigest(),
        'edges_bin_sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
        'sign_hypothesis': 'ACh +1; GABA -1; glutamate -1; all other/unknown 0',
        'sign_counts': {str(k): int((signs == k).sum()) for k in (-1, 0, 1)},
        'nt_labels': {str(k): int(v) for k, v in labels.value_counts().items()},
        'full_annotated_graph_preserved': True, 'raw_source_rows_preserved': True}
    return ids, edges, signs, motor, stim, report


def stimuli(seed, steps=120, amplitude=0.7):
    rng = np.random.default_rng(seed)
    levels = rng.uniform(0, amplitude, size=((steps + 14)//15, 2))
    drive = np.repeat(levels, 15, axis=0)[:steps]
    drive[:10] = 0
    drive[steps//2:steps//2+10, seed % 2] = 0
    return drive


def experiment(ids, edges, signs, motor, stim, gain=0.8):
    config = RateConfig(gain=gain)
    circuit = SparseCircuit(ids, edges['pre'], edges['post'], edges['count'], signs, config)
    unsigned = sparse.csr_matrix((edges['count'].astype(float), (edges['post'], edges['pre'])),
                                 shape=(len(ids), len(ids)))
    from_stim = np.asarray(unsigned[:, stim].sum(axis=1)).ravel()
    to_motor = np.asarray(unsigned[motor].sum(axis=0)).ravel()
    score = from_stim * to_motor
    score[np.concatenate((motor, stim))] = 0
    order = np.lexsort((ids, -score))
    region = np.sort(order[score[order] > 0][:64])
    if len(region) < 16:
        raise ValueError('Too few two-hop bridge candidates for prespecified experiment')
    del unsigned
    train_seeds, val_seeds, test_seeds = (0, 1), (11,), (101, 102, 103)
    train = [simulate(circuit, region, motor, stim, stimuli(s))['region'] for s in train_seeds]
    training = np.vstack(train)
    ranks = (2, 4, 8, 16)
    bases = {k: pod_basis(training, k) for k in ranks}
    validation = []
    for seed in val_seeds:
        drive = stimuli(seed)
        ref = simulate(circuit, region, motor, stim, drive)['motor']
        lesion = simulate(circuit, region, motor, stim, drive, 'lesion')['motor']
        for rank in ranks:
            candidate = simulate(circuit, region, motor, stim, drive, 'pod', bases[rank])['motor']
            validation.append({'seed': seed, 'rank': rank, **comparison(ref, lesion, candidate)})
    passing = [row['rank'] for row in validation if row['recovery_fraction'] is not None
               and row['recovery_fraction'] >= 0.95]
    chosen = min(passing) if passing else min(validation, key=lambda x: x['candidate_error'])['rank']
    basis = bases[chosen]
    random = np.linalg.qr(np.random.default_rng(991).normal(size=(len(region), chosen)))[0]
    trials = []
    for seed in test_seeds:
        # Higher amplitudes test extrapolation; no refitting/rank changes on test traces.
        drive = stimuli(seed, amplitude=1.1)
        ref = simulate(circuit, region, motor, stim, drive)
        les = simulate(circuit, region, motor, stim, drive, 'lesion')
        for mode, u in (('lesion', None), ('native', None), ('pod', basis),
                        ('shuffled', None), ('random_basis', random)):
            out = les if mode == 'lesion' else simulate(circuit, region, motor, stim,
                  drive, mode, u, shuffle_seed=seed)
            metric = comparison(ref['motor'], les['motor'], out['motor'])
            if mode == 'native' and not np.allclose(ref['motor'], out['motor'], atol=1e-12, rtol=1e-10):
                raise AssertionError('Native replacement altered full-graph response')
            trials.append({'seed': seed, 'mode': mode, **metric,
                'maximum_motor_error': float(np.max(np.abs(out['motor'] - ref['motor']))),
                'stimulus_sha256': out['stimulus_hash'], 'motor_trace_sha256': out['motor_hash']})
    internal = circuit.w[region][:, region]
    return {'model': 'contractive dimensionless signed rate model, not Brian2 LIF',
        'config': asdict(config), 'graph_sha256': circuit.fingerprint,
        'region_selection': 'top 64 anatomical two-hop DNge104->candidate->motor count products; exclude DN and motor; tie-break source ID',
        'region_body_ids': ids[region].tolist(), 'region_size': len(region),
        'internal_functional_edges': int(internal.nnz),
        'train_seeds': list(train_seeds), 'validation_seeds': list(val_seeds),
        'test_seeds': list(test_seeds), 'training_trace_sha256': array_hash(training),
        'selected_rank': chosen, 'latent_state_reduction_factor': len(region)/chosen,
        'selection_passed_validation': bool(passing), 'basis_sha256': array_hash(basis),
        'validation': validation, 'trials': trials,
        'limitations': ['Artificial direct DN drive, not sensory transduction or behavior.',
          'Unknown transmitter signs have zero functional weight; anatomical rows retained.',
          'POD retains internal operator and reconstructs all region ports; no speedup claim.',
          'Internal shuffle preserves degree/source-weight pairs, not weighted incoming strengths.',
          'One animal graph, one uncalibrated dynamics hypothesis; seeds are stimuli, not animals.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reuse', action='store_true'); p.add_argument('--gain', type=float, default=0.8)
    args = p.parse_args()
    started = time.monotonic()
    source = json.loads((args.data/'source-manifest.json').read_text()) if args.reuse else acquire(args.data)
    # Recheck all source bytes even when explicitly reusing a downloaded dataset.
    for key, item in source['files'].items():
        with (args.data/SOURCES[key]).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                raise ValueError('Source bytes do not match recorded provenance')
    ids, edges, signs, motor, stim, graph = import_graph(args.data)
    result = {'schema': 'circuit-bridge-malecns/v1', 'sources': source, 'graph': graph,
        'claims': {'real_connectome_loaded': True, 'full_annotated_graph_retained': True,
          'biological_validation': False, 'living_tissue_connected': False,
          'body_simulated': False, 'walking_demonstrated': False, 'upstream_lif_executed': False},
        'experiment': experiment(ids, edges, signs, motor, stim, args.gain),
        'software': {'python': platform.python_version(), 'numpy': np.__version__},
        'elapsed_seconds': time.monotonic() - started}
    write_json(args.out, result)


if __name__ == '__main__':
    main()
