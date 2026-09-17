"""Numerical checks of the explicit model-bound argument, not animal tests."""
import math
import numpy as np
import pytest
from scipy import sparse
from research.circuit_bridge.certificate import DefectMonitor, contraction
from research.circuit_bridge.connectome import SparseCircuit, RateConfig, simulate, internal_shuffle, comparison


@pytest.mark.parametrize('value', [-1, 1, 2, math.nan, math.inf, True])
def test_invalid_contraction(value):
    with pytest.raises(ValueError): DefectMonitor(value)


@pytest.mark.parametrize('value', [-1, math.nan, math.inf, True])
def test_invalid_defect_does_not_mutate_monitor(value):
    monitor = DefectMonitor(.8)
    with pytest.raises(ValueError): monitor.advance(value)
    assert monitor.steps == 0 and monitor.bound == 0


def test_geometric_series_and_initial_error():
    monitor = DefectMonitor(.75, bound=.4)
    for t in range(1, 12):
        expected = .4 * .75**t + .2 * (1-.75**t) / (1-.75)
        assert monitor.advance(.2) == pytest.approx(expected)
    assert monitor.steps == 11 and monitor.peak == monitor.bound


@pytest.mark.parametrize('gain,alpha', [(0,.2),(.4,.8),(.9,.1),(.8,1)])
def test_contraction(gain,alpha):
    assert contraction(gain,alpha) == pytest.approx(1-alpha+alpha*gain)


@pytest.mark.parametrize('seed', list(range(12)))
@pytest.mark.parametrize('mode', ['native', 'lesion', 'pod', 'shuffled'])
def test_actual_motor_error_is_below_online_bound(seed,mode):
    rng = np.random.default_rng(seed)
    n = 14
    post,pre = np.where(rng.random((n,n)) < .3)
    circuit = SparseCircuit(np.arange(100,100+n), pre, post,
        rng.integers(1,40,len(pre)), rng.choice([-1,0,1],n), RateConfig(gain=.85))
    region = np.arange(3,10)
    drive = rng.uniform(-.6,.6,(60,3))
    basis = np.linalg.qr(rng.normal(size=(7,2)))[0] if mode == 'pod' else None
    reference = simulate(circuit,region,np.arange(n),[0,2,7],drive)
    candidate = simulate(circuit,region,np.arange(n),[0,2,7],drive,mode,basis,seed)
    error = np.max(np.abs(candidate['motor']-reference['motor']),axis=1)
    assert np.all(error <= candidate['error_bound_inf'] + 2e-14)
    assert candidate['certificate']['uses_intact_reference'] is False
    assert candidate['certificate']['floating_point_rigorous'] is False
    assert reference['error_bound_inf'].max() == 0


@pytest.mark.parametrize('seed', list(range(10)))
def test_rewire_preserves_actual_degrees_and_source_weight_multisets(seed):
    rng = np.random.default_rng(63)
    a = rng.normal(size=(15,15)) * (rng.random((15,15)) < .22)
    original = sparse.csr_matrix(a)
    rewired = internal_shuffle(original,seed)
    assert rewired.nnz == original.nnz
    np.testing.assert_array_equal(np.diff(rewired.indptr),np.diff(original.indptr))
    np.testing.assert_array_equal(np.diff(rewired.tocsc().indptr),np.diff(original.tocsc().indptr))
    for j in range(15):
        np.testing.assert_array_equal(np.sort(rewired[:,j].data),np.sort(original[:,j].data))
    assert (internal_shuffle(original,seed) != rewired).nnz == 0
    assert (original != rewired).nnz > 0


def test_empty_traces_are_not_evidence():
    with pytest.raises(ValueError): comparison([],[],[])


def test_benchmark_end_to_end_synthetic_fixture():
    from research.circuit_bridge.benchmark import run
    # This small graph tests orchestration only. It is not a fly dataset.
    ids=np.arange(100,170); stim=np.array([0,1]); motor=np.array([68,69])
    rows=[]
    for r in range(2,66):
        rows.extend([(r%2,r,1+r%7),(r,68+r%2,1+r%5)])
        rows.append((r,2+(r-1)%64,1))
    pre,post,count=np.asarray(rows).T
    dtype=[('pre','<u4'),('post','<u4'),('count','<u4')]
    edges=np.zeros(len(rows),dtype=dtype)
    edges['pre'],edges['post'],edges['count']=pre,post,count
    traces={}
    result=run(ids,edges,np.ones(len(ids),dtype=np.int8),motor,stim,.8,traces)
    assert len(result['trials'])==30
    assert all(t['bound_check']['passed'] for t in result['trials'])
    assert {'training_region','basis','stress_training_region','stress_basis'}<=traces.keys()
    assert all(t['trace_key'] in traces for t in result['trials'])
    assert result['train_seeds']==[0,1] and result['test_seeds']==[101,102,103]
