"""Software fixtures, not animals or biological experiments."""
from copy import deepcopy
import numpy as np
import pytest
from scipy import sparse
from research.circuit_bridge.intervention_audit import (
    SOURCE_COMMIT, RUN_ID, GAINS, COHORTS, audit_result, blind_spot,
    isotropic_probe, number, orthonormal_basis, motor_channel_audit,
)
from research.circuit_bridge.connectome import ArrayEndpoint
from research.circuit_bridge.core import Request


def historical_fixture():
    result = {'schema':'circuit-bridge-certified/v1','status':'complete',
              'source_commit':SOURCE_COMMIT,'actions_run_id':RUN_ID,
              'experiments':{},'claims':{k:False for k in (
                  'biological_validation','body_simulated','living_tissue_connected',
                  'walking_demonstrated','upstream_lif_executed','whole_brain_speedup_demonstrated')}}
    for gain in GAINS:
        rows=[]
        for cohort,(seeds,modes) in COHORTS.items():
            for seed in seeds:
                for mode in modes:
                    error = {'lesion':1., 'native':0., 'pod':.01, 'shuffled':.03,
                             'random_basis':1.1, 'frozen_pod':.99,
                             'robust_pod':0., 'robust_random_basis':0.}[mode]
                    rows.append({'cohort':cohort,'seed':seed,'mode':mode,
                                 'candidate_error':error,'lesion_error':1.,
                                 'recovery_fraction':1-error})
        result['experiments'][gain]={'region_size':64,'selected_rank':4,
            'train_seeds':[0,1],'validation_seeds':[11],'test_seeds':[101,102,103],
            'stress':{'selected_rank':64,'train_seeds':[21,22],
                      'validation_seeds':[31],'test_seeds':[201,202,203]},'trials':rows}
    return result


@pytest.mark.parametrize('n,k', [(2,1),(8,1),(8,4),(8,7),(64,4),(64,63)])
@pytest.mark.parametrize('amplitude', [.001,.1,1.])
def test_constructed_drive_defeats_fixed_linear_decoder(n,k,amplitude):
    u=np.linalg.qr(np.random.default_rng(17).normal(size=(n,k)))[0]
    data=blind_spot(u, amplitude)
    w=data['witness']
    assert w['relative_region_l2_error']==pytest.approx(1,abs=1e-10)
    assert w['max_abs_drive']<=amplitude+1e-12
    # Independently execute the existing endpoint, not just a duplicate formula.
    ids=tuple(range(n))
    endpoint=ArrayEndpoint(sparse.eye(n,format='csr')*.3, ids, .25, u)
    request=Request(0,.01,tuple(map(str,ids)),tuple(w['drive']))
    output=np.asarray(endpoint.exchange(request).values)
    assert np.allclose(output,w['projected_next'],atol=1e-12)
    assert np.linalg.norm(output)<1e-10


@pytest.mark.parametrize('n', [1,2,8,64])
def test_full_rank_is_not_compression(n):
    report=blind_spot(np.eye(n))
    assert not report['nontrivial_compression'] and report['witness'] is None


@pytest.mark.parametrize('bad', [np.zeros((3,1)),np.ones((2,3)),np.array([[np.nan]]),
                              np.array([[np.inf]]),np.array([1.,0.]),np.eye(3)*2])
def test_bad_basis_rejected(bad):
    with pytest.raises(ValueError):orthonormal_basis(bad)


@pytest.mark.parametrize('bad', [True,False,float('nan'),float('inf'),'1',None])
def test_nonfinite_or_nonreal_rejected(bad):
    with pytest.raises(ValueError):number(bad)


@pytest.mark.parametrize('value', [0.,-1.,1.01])
def test_amplitude_contract(value):
    with pytest.raises(ValueError):blind_spot(np.eye(4)[:,:2],value)


def test_isotropic_energy_not_biology():
    u=np.linalg.qr(np.random.default_rng(1).normal(size=(64,4)))[0]
    r=isotropic_probe(u)
    assert r['isotropic_population_energy_loss_ratio']==.9375
    assert abs(r['observed_energy_loss_ratio']-.9375)<.006
    assert 'not animal' in r['statistical_unit']
    assert isotropic_probe(u)==r


def test_rank_full_isotope_has_zero_residual():
    assert isotropic_probe(np.eye(8))['observed_energy_loss_ratio']==0


def test_audit_flags_false_success_and_topology_insensitivity():
    source=historical_fixture(); original=deepcopy(source)
    report=audit_result(source)
    assert source==original
    assert report['cohort_count']==90
    for e in report['experiments'].values():
        assert e['gates']['nontrivial_compression_on_dn_inputs']
        assert not e['gates']['robust_nontrivial_stress_success']
        assert not e['gates']['frozen_basis_passes_independent_port_stress']
        assert e['gates']['shuffled_control_also_passes_dn_threshold']


@pytest.mark.parametrize('corruption', ['failed','source','run','schema','extra_gain','missing_gain',
    'missing_trial','duplicate_trial','unknown_trial','negative_error','zero_damage','nan',
    'wrong_recovery','native_error','leakage','duplicate_seed','rank0','rank65','rank_bool','biology'])
def test_historical_audit_fails_closed(corruption):
    r=historical_fixture();e=r['experiments']['0.8'];t=e['trials'][0]
    if corruption=='failed':r['status']='failed'
    elif corruption=='source':r['source_commit']='x'
    elif corruption=='run':r['actions_run_id']='0'
    elif corruption=='schema':r['schema']='other'
    elif corruption=='extra_gain':r['experiments']['0.7']=deepcopy(e)
    elif corruption=='missing_gain':r['experiments'].pop('0.5')
    elif corruption=='missing_trial':e['trials'].pop()
    elif corruption=='duplicate_trial':e['trials'].append(deepcopy(t))
    elif corruption=='unknown_trial':t['mode']='missing'
    elif corruption=='negative_error':t['candidate_error']=-1.
    elif corruption=='zero_damage':t['lesion_error']=0.
    elif corruption=='nan':t['candidate_error']=float('nan')
    elif corruption=='wrong_recovery':t['recovery_fraction']=.5
    elif corruption=='native_error':
        e['trials'][1]['candidate_error']=.1;e['trials'][1]['recovery_fraction']=.9
    elif corruption=='leakage':e['train_seeds']=[101]
    elif corruption=='duplicate_seed':e['train_seeds']=[0,0]
    elif corruption=='rank0':e['selected_rank']=0
    elif corruption=='rank65':e['stress']['selected_rank']=65
    elif corruption=='rank_bool':e['selected_rank']=True
    elif corruption=='biology':r['claims']['biological_validation']=True
    with pytest.raises(ValueError):audit_result(r)


def test_motor_channel_audit_detects_a_hidden_bad_channel():
    ref=np.ones((8,3)); les=np.zeros((8,3)); candidate=ref.copy()
    candidate[:,2]=0
    report=motor_channel_audit(ref,les,candidate,np.array([10,11,12]))
    for floor in report['effect_floor_sensitivity']:
        assert floor['eligible']==3 and floor['below_threshold']==1
        assert floor['minimum']==0 and floor['worst_motor_body_id']==12


def test_motor_channel_audit_no_lesion_is_not_success():
    ref=np.ones((8,3))
    report=motor_channel_audit(ref,ref,ref,np.arange(3))
    assert all(x['status']=='unmeasurable_lesion_effect' for x in report['effect_floor_sensitivity'])


@pytest.mark.parametrize('bad', ['shape','nan','ids','duplicate','threshold'])
def test_motor_channel_audit_rejects_bad_input(bad):
    ref=np.ones((8,3));les=np.zeros((8,3));candidate=ref.copy();ids=np.arange(3);threshold=.95
    if bad=='shape':candidate=candidate[:1]
    elif bad=='nan':candidate[0,0]=np.nan
    elif bad=='ids':ids=ids.astype(float)
    elif bad=='duplicate':ids[0]=ids[1]
    elif bad=='threshold':threshold=0
    with pytest.raises(ValueError):motor_channel_audit(ref,les,candidate,ids,threshold)
