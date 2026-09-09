import numpy as np
from organism_core.muscles import MuscleBank

def test_passive_geometry_does_not_create_active_force():
    muscle=MuscleBank([0.])
    for q in (-1.,0.,1.):
        np.testing.assert_array_equal(muscle.command([q],[4.]),[0.])
        assert not muscle.fiber_force.any()

def test_spike_precedes_force_and_antagonists_reverse_torque():
    positive=MuscleBank([0.]);negative=MuscleBank([0.])
    assert positive.command([0.],[0.])[0]==0
    positive.observe([[1,0]]);negative.observe([[0,1]])
    tp=positive.command([0.],[0.])[0];tn=negative.command([0.],[0.])[0]
    assert tp>0 and tp==-tn
    assert positive.activation[0,0]<1

def test_pending_spike_and_activation_restore_exactly():
    a=MuscleBank([.2,.3]);a.observe([[1,2],[3,0]])
    state=a.state();b=MuscleBank([.2,.3]);b.restore(state)
    for _ in range(20):
        np.testing.assert_array_equal(a.command([.1,.4],[1.,-2.]),b.command([.1,.4],[1.,-2.]))

def test_force_velocity_and_length_change_force_without_angle_tracking():
    a=MuscleBank([0.]);a.observe([[20,0]]);a.command([0.],[0.]);state=a.state()
    resting=a.command([0.],[0.])[0]
    a.restore(state);shortening=a.command([0.],[20.])[0]
    a.restore(state);long=a.command([4.],[0.])[0]
    assert 0<=shortening<resting and 0<long<resting
