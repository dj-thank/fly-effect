import numpy as np
from analyze_walk import sustained_onsets

def test_contact_chatter_is_not_counted_as_steps():
    chatter=np.tile(np.arange(100)%2,(6,1)).T.astype(bool)
    assert not sustained_onsets(chatter).any()
    stable=np.tile(np.repeat([False,True,False,True,False,True],10),(6,1)).T
    np.testing.assert_array_equal(sustained_onsets(stable),np.full(6,3))
