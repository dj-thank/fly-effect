import numpy as np
import pytest
mj=pytest.importorskip('mujoco')
from organism_core.mechanics import DEMO_XML,pulling_force
from organism_core.transmission import SiteTendonTransmission,audit_transmission,compare_columns


@pytest.mark.parametrize('jacobian',[mj.mjtJacobian.mjJAC_DENSE,mj.mjtJacobian.mjJAC_SPARSE])
@pytest.mark.parametrize('angle',[-.4,0.,.3])
def test_native_transmission_matches_all_oracles_without_state_mutation(jacobian,angle):
    model=mj.MjModel.from_xml_string(DEMO_XML); model.opt.jacobian=jacobian
    data=mj.MjData(model); data.qpos[-1]=angle; data.qpos[3:7]=[.5,.5,.5,.5]
    data.qvel[:]=np.linspace(-.3,.3,model.nv); mj.mj_forward(model,data)
    trans=SiteTendonTransmission(model); before=data.qpos.copy(); applied=data.qfrc_applied.copy()
    result,matrices=audit_transmission(model,data)
    assert result['passed'] and result['reversed_observable_columns_rejected']
    np.testing.assert_array_equal(before,data.qpos)
    np.testing.assert_array_equal(applied,data.qfrc_applied)
    np.testing.assert_allclose(trans.force(data,[2.]),pulling_force(model,data,trans.paths,[2.]),atol=1e-12)
    np.testing.assert_allclose(matrices['native'][:6],0.,atol=1e-12)
    assert trans.force(data,[2.])@data.qvel==pytest.approx(-2*data.ten_velocity[0],abs=1e-10)


def test_columnwise_audit_cannot_hide_cancellation_or_permutation():
    reference=np.array([[1.,-1.],[2.,-2.]])
    wrong=reference[:,::-1]
    assert np.allclose(wrong@np.ones(2),reference@np.ones(2))
    assert not compare_columns(wrong,reference)['passed']
    assert not compare_columns(-reference,reference)['passed']


@pytest.mark.parametrize('tension',[[],[1.,2.],[-1.],[np.nan],[np.inf]])
def test_invalid_tension_rejected(tension):
    m=mj.MjModel.from_xml_string(DEMO_XML); d=mj.MjData(m); mj.mj_forward(m,d)
    with pytest.raises(ValueError):SiteTendonTransmission(m).force(d,tension)


def test_degenerate_segment_rejected_by_fast_path():
    m=mj.MjModel.from_xml_string(DEMO_XML); d=mj.MjData(m); mj.mj_forward(m,d)
    transmission=SiteTendonTransmission(m); d.site_xpos[1]=d.site_xpos[0]
    with pytest.raises(ValueError,match='degenerate'):transmission.force(d,[0.])


def test_zero_tension_is_zero_force():
    m=mj.MjModel.from_xml_string(DEMO_XML); d=mj.MjData(m); mj.mj_forward(m,d)
    np.testing.assert_array_equal(SiteTendonTransmission(m).force(d,[0]),np.zeros(m.nv))


def test_shared_intermediate_site_multiple_tendons():
    xml=DEMO_XML.replace('<site name="origin" pos="0 1 0"/>',
        '<site name="origin" pos="0 1 0"/><site name="via" pos=".1 .5 0"/><site name="other" pos="0 -1 0"/>')
    xml=xml.replace('<site site="origin"/><site site="insertion"/>',
        '<site site="origin"/><site site="via"/><site site="insertion"/>')
    xml=xml.replace('</tendon>', '<spatial name="opposite"><site site="other"/><site site="insertion"/></spatial></tendon>')
    m=mj.MjModel.from_xml_string(xml); d=mj.MjData(m); mj.mj_forward(m,d)
    trans=SiteTendonTransmission(m)
    assert audit_transmission(m,d)[0]['passed']
    np.testing.assert_allclose(trans.force(d,[2.,3.]),pulling_force(m,d,trans.paths,[2.,3.]),atol=1e-12)
