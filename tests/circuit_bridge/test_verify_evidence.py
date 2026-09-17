"""Synthetic payloads only: test rejection of invalid evidence, not biological data."""
from copy import deepcopy
import numpy as np
import pytest
from research.circuit_bridge.verify_evidence import audit_payload, fingerprint, metrics, MODES


def payload():
    # Fake anatomical metadata belongs only to this in-memory software fixture.
    training = np.array([[1.,0.],[0.,2.],[1.,1.]])
    _,_,vh=np.linalg.svd(training,full_matrices=False)
    basis=vh[:2].T
    reference=np.ones((4,815))*.1
    lesion=reference*.5
    drive=np.ones((4,2))*.2
    region=np.ones((4,2))*.1
    traces={'training':{'basis':basis,'training':training,'region_ids':np.array([1,2]),
            'motor_ids':np.arange(10,825),'stimulus_ids':np.array([12781,556329])},
            '101-intact':{'motor':reference,'region':region,'drive':drive}}
    trials=[]
    for mode in MODES:
        value=lesion if mode=='lesion' else reference.copy()
        r=np.zeros_like(region) if mode=='lesion' else region.copy()
        traces[f'101-{mode}']={'motor':value,'region':r,'drive':drive.copy()}
        trials.append({'seed':101,'mode':mode,**metrics(reference,lesion,value),
            'stimulus_sha256':fingerprint(drive),'motor_trace_sha256':fingerprint(value),
            'status':'model_response_only'})
    result={'schema':'circuit-bridge-malecns/v1',
        'claims':{key:False for key in ('biological_validation','living_tissue_connected',
                    'body_simulated','walking_demonstrated','upstream_lif_executed')},
        'graph':{'neurons':166700,'connection_rows':25582938,'motor_neurons':815,
                 'stimulus_body_ids':[12781,556329],
                 'upstream_byte_parity':{key:True for key in ('body_ids.npy','edges.bin',
                        'motor_indices.npy','glutamate_inhibitory_hypothesis_signs.npy')}},
        'experiment':{'train_seeds':[0,1],'validation_seeds':[11],'test_seeds':[101],
                'region_size':2,'selected_rank':2,'region_body_ids':[1,2],
                'basis_sha256':fingerprint(basis),'training_trace_sha256':fingerprint(training),
                'validation':[{'seed':11,'rank':2,'recovery_fraction':1.,'candidate_error':0.}],
                'selection_passed_validation':True,'trials':trials}}
    execution={'result_present':True,'source_commit':'a'*40,'actions_run_id':'123'}
    return result,execution,traces,'a'*40


def test_valid_synthetic_payload_checks_all_comparisons():
    out=audit_payload(*payload())
    assert out['verified'] and out['metric_recomputations']==5
    assert out['aggregate']['pod']['recovery_min']==1


@pytest.mark.parametrize('claim',['biological_validation','living_tissue_connected',
                    'body_simulated','walking_demonstrated','upstream_lif_executed'])
def test_exaggerated_claim_rejected(claim):
    r,e,t,c=payload();r['claims'][claim]=True
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


@pytest.mark.parametrize('field',['candidate_error','recovery_fraction','maximum_motor_error'])
def test_invented_metric_rejected(field):
    r,e,t,c=payload();r['experiment']['trials'][2][field]+=.1
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_successful_job_without_result_is_not_evidence():
    r,e,t,c=payload();e['result_present']=False
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_changed_trace_rejected():
    r,e,t,c=payload();t['101-pod']['motor'][0,0]+=.01
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_changed_stimulus_rejected():
    r,e,t,c=payload();t['101-pod']['drive'][1,0]+=.1
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_train_test_leakage_rejected():
    r,e,t,c=payload();r['experiment']['test_seeds']=[0]
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_missing_negative_control_rejected():
    r,e,t,c=payload();r['experiment']['trials'].pop()
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_duplicate_condition_rejected():
    r,e,t,c=payload();r['experiment']['trials'][-1]=deepcopy(r['experiment']['trials'][0])
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_incomplete_anatomy_rejected():
    r,e,t,c=payload();r['graph']['neurons']=166699
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_source_version_mismatch_rejected():
    r,e,t,c=payload();e['source_commit']='b'*40
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_training_basis_tampering_rejected():
    r,e,t,c=payload();t['training']['basis'][0,0]+=.01
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_nonzero_lesioned_region_rejected():
    r,e,t,c=payload();t['101-lesion']['region'][0,0]=.1
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_bad_native_region_parity_even_when_motor_identical_rejected():
    r,e,t,c=payload();t['101-native']['region'][0,0]+=.1
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_native_motor_corruption_rejected_even_with_recomputed_metrics():
    r,e,t,c=payload();t['101-native']['motor']*=.9
    row=next(x for x in r['experiment']['trials'] if x['mode']=='native')
    row.update(metrics(t['101-intact']['motor'],t['101-lesion']['motor'],t['101-native']['motor']))
    row['motor_trace_sha256']=fingerprint(t['101-native']['motor'])
    with pytest.raises(ValueError):audit_payload(r,e,t,c)


def test_negative_recovery_not_clamped():
    reference=np.ones((2,3));lesion=np.zeros((2,3));candidate=reference*3
    assert metrics(reference,lesion,candidate)['recovery_fraction']==-1


def test_no_damage_metric_is_null():
    z=np.zeros((2,3))
    assert metrics(z,z,z)['recovery_fraction'] is None
