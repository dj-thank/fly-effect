"""Small synthetic fixtures test software, never stand in for biological data."""
import numpy as np
import pytest
from scipy import sparse
from research.circuit_bridge.connectome import (SparseCircuit, RateConfig, ArrayEndpoint,
    simulate, pod_basis, comparison, internal_shuffle, indices)
from research.circuit_bridge.core import Request, Boundary, BoundaryError


def fixture():
    return SparseCircuit(np.arange(10, 16), np.array([0,1,2,3,1,2,3,4]),
        np.array([1,2,3,1,4,4,5,5]), np.array([2,3,2,1,4,2,1,3]),
        np.array([1,1,-1,1,1,0]))


def run(c, mode='intact', basis=None):
    stimulus = np.linspace(0, 0.9, 50).reshape(-1,1)
    return simulate(c, [1,2,3], [4,5], [0], stimulus, mode, basis)


@pytest.mark.parametrize('gain', [0,1,2,-1,np.nan,np.inf,True])
def test_bad_gain(gain):
    with pytest.raises(ValueError): RateConfig(gain=gain)


@pytest.mark.parametrize('values', [[], [1,1], [-1], [6], [1.0], [[1]], [True]])
def test_bad_indices(values):
    with pytest.raises(ValueError): indices(values, 6, 'test')


def test_native_parity_and_no_graph_mutation():
    c = fixture(); original = c.w.copy(); fp = c.fingerprint
    ref, sham = run(c), run(c, 'native')
    np.testing.assert_allclose(ref['motor'], sham['motor'], atol=1e-15)
    np.testing.assert_allclose(ref['region'], sham['region'], atol=1e-15)
    assert (original != c.w).nnz == 0 and c.fingerprint == fp
    assert ref['stimulus_hash'] == sham['stimulus_hash']
    with pytest.raises(ValueError): c.w.data[0] = 2


def test_lesion_and_full_rank_recovery():
    c = fixture(); ref = run(c); lesion = run(c, 'lesion')
    np.testing.assert_equal(lesion['region'], np.zeros((50,3)))
    assert np.linalg.norm(ref['motor'] - lesion['motor']) > 0
    full = run(c, 'pod', np.eye(3))
    assert comparison(ref['motor'], lesion['motor'], full['motor'])['recovery_fraction'] == pytest.approx(1)


def test_pod_orthogonality_and_unseen_input():
    c = fixture(); train = run(c)['region']
    basis = pod_basis(train, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
    drive = np.random.default_rng(8).uniform(0,.8,size=(40,1))
    result = simulate(c, [1,2,3], [4,5], [0], drive, 'pod', basis)
    assert np.isfinite(result['motor']).all()


def test_ports_account_for_internal_and_external_input():
    c = fixture(); r, inner, incoming = c.ports([1,2,3])
    state = np.linspace(-.3,.3,6)
    np.testing.assert_allclose(c.w[r] @ state, inner @ state[r] + incoming @ state, atol=1e-15)
    assert incoming[:,r].nnz == 0


def test_direct_stimulation_inside_region_is_not_lost():
    c = fixture(); drive = np.ones((25,1))*.2
    a = simulate(c,[1,2,3],[4,5],[2],drive)
    b = simulate(c,[1,2,3],[4,5],[2],drive,'native')
    np.testing.assert_allclose(a['motor'],b['motor'],atol=1e-15)


def test_shuffle_preserves_source_weight_pairs_and_post_degree():
    inner = sparse.csr_matrix(np.array([[.1,.2,.3],[.4,0,.5],[.2,.8,0]]))
    before = inner.tocoo(); after = internal_shuffle(inner,41).tocoo()
    # This fixture/seed has no merged duplicate pair.
    assert np.sum(after.data) == pytest.approx(np.sum(before.data))
    assert internal_shuffle(inner,41).shape == inner.shape
    c = fixture(); original = c.w.copy(); run(c,'shuffled')
    assert (original != c.w).nnz == 0


@pytest.mark.parametrize('rank',[0,-1,4,True])
def test_bad_pod_rank(rank):
    with pytest.raises(ValueError): pod_basis(np.ones((20,3)),rank)


def test_no_effect_has_no_recovery():
    out = comparison(np.zeros((2,3)),np.zeros((2,3)),np.ones((2,3)))
    assert out['recovery_fraction'] is None
    assert out['relative_reference_error'] is None


def test_negative_recovery_is_not_clipped():
    out = comparison(np.ones((2,3)),np.zeros((2,3)),np.ones((2,3))*3)
    assert out['recovery_fraction'] < 0


def test_unknown_transmitter_keeps_anatomical_identity_but_no_current():
    c = SparseCircuit(np.array([10,11]),np.array([0]),np.array([1]),np.array([5]),np.array([0,1]))
    assert c.w.nnz == 1 and c.w.data[0] == 0


@pytest.mark.parametrize('basis',[np.ones((3,2)),np.ones((2,1)),np.full((3,1),np.nan)])
def test_bad_basis(basis):
    with pytest.raises(ValueError): ArrayEndpoint(sparse.eye(3,format='csr'),[1,2,3],.25,basis)


def test_wrong_port_identity_latches_boundary():
    endpoint = ArrayEndpoint(sparse.eye(2,format='csr'),[1,2],.25)
    boundary = Boundary(endpoint,['2','1'],.01)
    with pytest.raises(BoundaryError): boundary.exchange(0,[0,0])
    assert boundary.failed


@pytest.mark.parametrize('bad', [np.array([0]),np.array([-2]),np.array([1.2])])
def test_invalid_counts(bad):
    with pytest.raises(ValueError): SparseCircuit(np.array([10,11]),np.array([0]),np.array([1]),bad,np.array([1,1]))


def test_condition_order_and_repetition_are_independent():
    c = fixture(); a = run(c); run(c,'lesion'); run(c,'shuffled'); b = run(c)
    assert a['motor_hash'] == b['motor_hash'] and a['final_hash'] == b['final_hash']


def test_nonfinite_drive_rejected():
    with pytest.raises(ValueError): simulate(fixture(),[1],[4],[0],[[np.nan]])
