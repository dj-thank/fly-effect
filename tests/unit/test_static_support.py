import numpy as np
import pytest
from organism_core.static_support import solve_support
pytest.importorskip('scipy')


def test_weight_supported_by_compressive_ground_only():
    result=solve_support([[0]],[[0,0,1]],[1],[1],[.5])
    assert result['feasible'] and not result['walking_claimed']
    np.testing.assert_allclose(result['contact_forces_native'],[[0,0,1]])
    assert result['tensions_native']==[0]


def test_contact_cannot_pull_the_ground():
    result=solve_support([[0]],[[0,0,1]],[-1],[1],[.5])
    assert not result['feasible'] and result['solver_status']==2


@pytest.mark.parametrize('target,feasible',[(.5,True),(1.,True),(1.1,False),(-.1,False)])
def test_tendon_tension_is_bounded_and_cannot_push(target,feasible):
    result=solve_support([[0],[1]],[[0,0,1],[0,0,0]],[1,target],[1],[.5])
    assert result['feasible']==feasible
    if feasible: assert result['tensions_native'][0]==pytest.approx(target)


@pytest.mark.parametrize('horizontal,feasible',[(.4,True),(.6,False)])
def test_contact_friction_limit(horizontal,feasible):
    result=solve_support([[0],[0]],[[0,0,1],[1,0,0]],[1,horizontal],[1],[.5])
    assert result['feasible']==feasible


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,0.,-1.])
def test_invalid_maximum_tension(value):
    with pytest.raises(ValueError):solve_support([[0]],[[0,0,1]],[1],[value],[.5])


def test_invalid_dimensions_do_not_reach_solver():
    with pytest.raises(ValueError):solve_support([[0]],[[1,0]],[1],[1],[.5])


def test_no_foot_contact_cannot_be_claimed_as_standing():
    with pytest.raises(ValueError):solve_support([[0]],np.empty((1,0)),[1],[1],[])


def test_inputs_unchanged_and_residual_is_checked():
    T=np.array([[0.],[1.]]); C=np.array([[0.,0.,1.],[0.,0.,0.]])
    b=np.array([1.,.25]); f=np.array([1.]); mu=np.array([.5])
    old=[x.copy() for x in (T,C,b,f,mu)]
    result=solve_support(T,C,b,f,mu)
    for x,y in zip((T,C,b,f,mu),old):np.testing.assert_array_equal(x,y)
    assert result['feasible'] and result['maximum_scaled_residual']<1e-10


def test_infeasibility_has_residual_diagnostic_not_success():
    result=solve_support([[0]],[[0,0,1]],[-1],[1],[.5])
    assert result['status']=='infeasible_under_declared_constraints'
    nearest=result['nearest_balance']
    assert not result['feasible'] and not nearest['is_feasible_solution']
    assert nearest['minimum_scaled_balance_error']==pytest.approx(1.)
    assert nearest['residual_native']==pytest.approx([1.])


def test_solver_timeout_is_not_reported_as_infeasibility(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize,'linprog',lambda *a,**k:
        SimpleNamespace(status=1,message='time limit',success=False))
    result=solve_support([[0]],[[0,0,1]],[1],[1],[.5])
    assert result['status']=='inconclusive' and not result['feasible']
    assert 'nearest_balance' not in result


@pytest.mark.parametrize('values',[[np.nan,0.,0.,1.],[0.,0.,0.],[0.,0.,0.,np.inf]])
def test_invalid_solver_solution_is_not_feasible(monkeypatch,values):
    from types import SimpleNamespace
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize,'linprog',lambda *a,**k:
        SimpleNamespace(status=0,message='mock success',success=True,x=np.array(values)))
    result=solve_support([[0]],[[0,0,1]],[1],[1],[.5])
    assert result['status']=='invalid_solver_solution' and not result['feasible']
