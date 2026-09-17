"""Sparse, port-preserving interventions on real anatomical graphs.

Dynamics are an explicit dimensionless contractive rate hypothesis, NOT the
upstream Brian2 LIF model or biological validation. No graph is relabelled as
synthetic and no upstream acceptance gate is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import sparse
from .core import Boundary, Request, Response, digest


def array_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def indices(values, n: int, name: str) -> np.ndarray:
    a = np.asarray(values)
    if (a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a)
            or len(np.unique(a)) != len(a) or np.any(a < 0) or np.any(a >= n)):
        raise ValueError(f'{name}: unique in-range integer indices required')
    return a.astype(np.int64)


@dataclass(frozen=True)
class RateConfig:
    gain: float = 0.8
    alpha: float = 0.25
    dt_s: float = 0.01

    def __post_init__(self):
        if (any(isinstance(x, bool) or not np.isfinite(x)
                for x in (self.gain, self.alpha, self.dt_s))
                or not 0 < self.gain < 1 or not 0 < self.alpha <= 1
                or not 0 < self.dt_s <= 1):
            raise ValueError('Require 0<gain<1, 0<alpha<=1, 0<dt_s<=1')


class SparseCircuit:
    """W[post, pre]; normalize ONCE before applying any intervention."""
    def __init__(self, ids, pre, post, counts, signs, config=RateConfig()):
        self.ids = np.asarray(ids).copy()
        n = len(self.ids)
        if (self.ids.ndim != 1 or self.ids.dtype.kind not in 'iu'
                or not n or len(np.unique(self.ids)) != n):
            raise ValueError('Unique integer source IDs required')
        pre, post, counts, signs = map(np.asarray, (pre, post, counts, signs))
        if (pre.ndim != 1 or post.shape != pre.shape or counts.shape != pre.shape
                or pre.dtype.kind not in 'iu' or post.dtype.kind not in 'iu'
                or counts.dtype.kind not in 'iu' or np.any(counts <= 0)
                or np.any(pre < 0) or np.any(pre >= n)
                or np.any(post < 0) or np.any(post >= n)
                or signs.shape != (n,) or not np.isin(signs, [-1, 0, 1]).all()):
            raise ValueError('Invalid anatomical edges or explicit sign hypothesis')
        self.config = config
        # Unknown signs retain anatomical edges but have zero functional weight.
        denominator = np.bincount(post.astype(np.int64), weights=counts, minlength=n)
        weight = config.gain * counts.astype(float) * signs[pre] / np.maximum(denominator[post], 1)
        self.w = sparse.csr_matrix((weight, (post, pre)), shape=(n, n))
        self.w.sum_duplicates(); self.w.sort_indices()
        self.fingerprint = digest({'ids': array_hash(self.ids), 'pre': array_hash(pre),
            'post': array_hash(post), 'counts': array_hash(counts), 'signs': array_hash(signs)})
        self.ids.flags.writeable = False
        for a in (self.w.data, self.w.indices, self.w.indptr):
            a.flags.writeable = False

    def ports(self, region):
        r = indices(region, len(self.ids), 'region')
        inside = self.w[r][:, r].tocsr()
        incoming = self.w[r].tolil()
        incoming[:, r] = 0
        incoming = incoming.tocsr(); incoming.eliminate_zeros()
        return r, inside, incoming


class ArrayEndpoint:
    """Native or POD/Galerkin endpoint; keeps only rank-k latent state in POD mode.

    The learned basis is fitted on training traces only. It retains a copy of
    the internal anatomical operator: this is model-informed order reduction,
    NOT a black-box learned replacement or a reduction in anatomical neurons.
    """
    def __init__(self, internal, ids, alpha, basis=None):
        self.internal = internal
        self.ids = tuple(str(int(x)) for x in ids)
        self.alpha = alpha
        self.basis = None if basis is None else np.asarray(basis, dtype=float).copy()
        n = len(self.ids)
        if internal.shape != (n, n) or not np.isfinite(internal.data).all():
            raise ValueError('Invalid internal operator')
        if self.basis is not None:
            k = self.basis.shape[1] if self.basis.ndim == 2 else 0
            if (self.basis.shape != (n, k) or not 1 <= k <= n
                    or not np.isfinite(self.basis).all()
                    or not np.allclose(self.basis.T @ self.basis, np.eye(k), atol=1e-10, rtol=0)):
                raise ValueError('POD basis must have orthonormal finite columns')
        self.state = np.zeros(n if self.basis is None else self.basis.shape[1])

    def exchange(self, request: Request) -> Response:
        if request.channel_ids != self.ids:
            raise ValueError('Port identity/order mismatch')
        old = self.state if self.basis is None else self.basis @ self.state
        new = ((1 - self.alpha) * old + self.alpha *
               np.tanh(self.internal @ old + np.asarray(request.drive)))
        if self.basis is None:
            self.state = new
        else:
            self.state = self.basis.T @ new
            new = self.basis @ self.state
        # Do not clip a defective surrogate into passing the boundary contract.
        return Response.for_request(request, new.tolist())


def pod_basis(traces, rank: int) -> np.ndarray:
    a = np.asarray(traces, dtype=float)
    if (a.ndim != 2 or not np.isfinite(a).all() or type(rank) is not int
            or not 1 <= rank <= min(a.shape) or np.linalg.norm(a) <= 1e-14):
        raise ValueError('Nonzero training-only trace matrix and valid rank required')
    _, _, vh = np.linalg.svd(a, full_matrices=False)
    return vh[:rank].T


def internal_shuffle(internal, seed: int):
    """Permute internal posts, preserving each pre/weight pair and post degree.

    Does NOT preserve weighted post strength; report this limitation. No
    renormalization, port rewiring or use of held-out responses is allowed.
    """
    coo = internal.tocoo(copy=True)
    rows = np.random.default_rng(seed).permutation(coo.row)
    out = sparse.csr_matrix((coo.data, (rows, coo.col)), shape=internal.shape)
    out.sum_duplicates(); out.sort_indices()
    return out


def simulate(circuit, region, motor, stimulus_indices, stimulus, mode='intact',
             basis=None, shuffle_seed=0, capture_region=True):
    """Full graph remains in circuit; only region state is overridden.

    External stimulus is open-loop artificial neural drive. Recurrent neural
    feedback remains live in ALL conditions. No body/behavior is simulated.
    """
    r, inside, incoming = circuit.ports(region)
    motor = indices(motor, len(circuit.ids), 'motor')
    stim_idx = indices(stimulus_indices, len(circuit.ids), 'stimulus')
    stim = np.asarray(stimulus, dtype=float)
    if (stim.ndim != 2 or stim.shape[1] != len(stim_idx) or not len(stim)
            or not np.isfinite(stim).all()):
        raise ValueError('Invalid stimulus')
    if mode not in ('intact', 'lesion', 'native', 'pod', 'shuffled', 'random_basis'):
        raise ValueError('Unknown intervention')
    if mode in ('pod', 'random_basis') and basis is None:
        raise ValueError('Compressed mode requires a training-frozen basis')
    conf = circuit.config
    boundary = None
    if mode in ('native', 'pod', 'random_basis', 'shuffled'):
        inner = internal_shuffle(inside, shuffle_seed) if mode == 'shuffled' else inside
        endpoint = ArrayEndpoint(inner, circuit.ids[r], conf.alpha,
                                 basis if mode in ('pod', 'random_basis') else None)
        boundary = Boundary(endpoint, tuple(str(int(x)) for x in circuit.ids[r]), conf.dt_s,
                            deadline_s=30)
    state = np.zeros(len(circuit.ids))
    motor_trace = np.empty((len(stim), len(motor)))
    region_trace = np.empty((len(stim), len(r))) if capture_region else None
    region_stim = np.zeros((len(stim), len(r)))
    lookup = {int(i): j for j, i in enumerate(r)}
    for j, idx in enumerate(stim_idx):
        if int(idx) in lookup:
            region_stim[:, lookup[int(idx)]] = stim[:, j]
    for step, drive in enumerate(stim):
        total = circuit.w @ state
        total[stim_idx] += drive
        nxt = (1 - conf.alpha) * state + conf.alpha * np.tanh(total)
        if mode == 'lesion':
            nxt[r] = 0
        elif boundary is not None:
            port_drive = incoming @ state + region_stim[step]
            nxt[r] = boundary.exchange(step, port_drive.tolist())
            # Long full-CNS runs retain compact trace hashes, not millions of dictionaries.
            boundary.records.clear()
        if not np.isfinite(nxt).all() or np.max(np.abs(nxt)) > 1 + 1e-12:
            raise ValueError('Nonfinite or out-of-contract circuit state')
        state = nxt
        motor_trace[step] = state[motor]
        if region_trace is not None:
            region_trace[step] = state[r]
    return {'motor': motor_trace, 'region': region_trace,
            'final_hash': array_hash(state), 'stimulus_hash': array_hash(stim),
            'motor_hash': array_hash(motor_trace)}


def comparison(reference, lesion, candidate):
    arrays = [np.asarray(x, dtype=float) for x in (reference, lesion, candidate)]
    if any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Finite, shape-matched traces required')
    ref, les, cand = arrays
    norm = float(np.linalg.norm(ref))
    damage = float(np.linalg.norm(les - ref))
    error = float(np.linalg.norm(cand - ref))
    measurable = damage > max(1e-12, norm * 1e-8)
    return {'reference_norm': norm, 'lesion_error': damage, 'candidate_error': error,
            'relative_reference_error': error / norm if norm > 1e-12 else None,
            'recovery_fraction': 1 - error / damage if measurable else None,
            'status': 'model_response_only' if measurable else 'unmeasurable_lesion_effect'}
