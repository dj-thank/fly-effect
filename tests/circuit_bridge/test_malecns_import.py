import pytest

def test_annotation_selection_preserves_source_order_and_all_statuses():
    import pandas as pd
    from research.circuit_bridge.malecns import select_neuronal_domain
    source = pd.DataFrame({'bodyId':[70,10,80,30],
        'superclass':['vnc_motor', None, 'central', ''],
        'status':['Cropped','Traced','Untraced','Traced']})
    out = select_neuronal_domain(source)
    assert out.bodyId.tolist() == [70,80]
    assert out.status.tolist() == ['Cropped','Untraced']
    assert len(source) == 4


@pytest.mark.parametrize('bad_ids', [[1,1],[0,2],[-1,2],[1.0,2.0]])
def test_annotation_primary_key_rejected(bad_ids):
    import pandas as pd
    from research.circuit_bridge.malecns import select_neuronal_domain
    with pytest.raises(ValueError):
        select_neuronal_domain(pd.DataFrame({'bodyId':bad_ids,'superclass':['a','b']}))
