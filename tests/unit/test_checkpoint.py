import numpy as np
import pytest
from organism_core.checkpoint import save,load

def test_nested_arrays_preserve_delays_and_rng(tmp_path):
    state={'queue':(4,np.array([[1,8],[4,6]],dtype=np.int32)),
           'rng':np.random.default_rng(5).bit_generator.state,'empty':np.array([],dtype=np.uint32)}
    receipt=save(tmp_path/'state.npz',state,{'dataset':'fixture'})
    actual=load(tmp_path/'state.npz',{'dataset':'fixture'},receipt['sha256'])
    assert actual['queue'][0]==4
    np.testing.assert_array_equal(actual['queue'][1],state['queue'][1])
    assert actual['rng']==state['rng'] and actual['empty'].dtype==np.uint32
    with pytest.raises(ValueError,match='identity'):load(tmp_path/'state.npz',{'dataset':'wrong'})

def test_object_array_refused(tmp_path):
    with pytest.raises(TypeError):save(tmp_path/'bad.npz',{'x':np.array([object()])},{})
