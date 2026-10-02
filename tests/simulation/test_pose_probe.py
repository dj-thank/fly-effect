import time
import numpy as np
import pytest
mj = pytest.importorskip('mujoco')
from organism_core.pose_probe import first_contact_height, ground_contacts

XML = '''<mujoco><worldbody><geom name="ground_plane" type="plane" size="2 2 .1"/>
<body pos="0 0 .5"><freejoint/><geom name="organism/lf_tarsus1" type="sphere" size=".1" mass="1"/>
</body></worldbody></mujoco>'''


def test_first_contact_bracket_recovers_analytic_sphere_height_and_preserves_qpos():
    m=mj.MjModel.from_xml_string(XML);d=mj.MjData(m);q=d.qpos.copy()
    height=first_contact_height(m,d,q,deadline=time.monotonic()+10)
    assert height==pytest.approx(.1,abs=2e-7)
    np.testing.assert_array_equal(q,m.qpos0)


def test_ground_loads_record_positive_contact_force():
    m=mj.MjModel.from_xml_string(XML);d=mj.MjData(m);d.qpos[2]=.099
    mj.mj_forward(m,d)
    rows=ground_contacts(m,d)
    assert len(rows)==1 and rows[0]['foot']
    assert rows[0]['normal_force_native']>0 and rows[0]['world_force_native'][2]>0
    # Instantaneous penetration is not an equilibrium: don't assert exact mg.


def test_pose_probe_honors_wall_budget():
    m=mj.MjModel.from_xml_string(XML);d=mj.MjData(m)
    with pytest.raises(TimeoutError):first_contact_height(m,d,d.qpos,deadline=time.monotonic()-1)


def test_pose_probe_rejects_unbracketed_geometry():
    m=mj.MjModel.from_xml_string(XML.replace('pos="0 0 .5"','pos="0 0 9"'));d=mj.MjData(m)
    with pytest.raises(ValueError,match='bracketed'):
        first_contact_height(m,d,d.qpos,deadline=time.monotonic()+10)
