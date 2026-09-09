from types import SimpleNamespace
import pytest
from organism_core.contacts import is_active_contact


@pytest.mark.parametrize('address,exclude,active', [
    (0,0,True),(12,0,True),(-1,0,False),(-1,1,False),
    (0,1,False),(0,2,False),(0,3,False),(0,4,False)])
def test_constraint_membership_not_surface_distance(address,exclude,active):
    # No dist field: distance is deliberately not the criterion.
    assert is_active_contact(SimpleNamespace(efc_address=address,exclude=exclude)) == active
