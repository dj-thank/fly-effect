"""Real MuJoCo root controls; no FlyGym download or neural dataset required."""
import numpy as np
import pytest
mj = pytest.importorskip('mujoco')
from organism_core.root_ab import analytic_control, audit_models, derive_xml

XML = '''<mujoco><option timestep=".0001" gravity="0 0 -9.81"/>
<worldbody><geom name="ground_plane" type="plane" size="10 10 .1"/>
<body name="organism" pos="0 0 2">
<joint name="root" type="free" stiffness=".4" damping=".02" armature="1e-6"/>
<geom type="sphere" size=".1" mass="1"/><site name="origin" pos=".1 0 0"/>
<body name="leg" pos="0 0 -.3"><joint name="limb" stiffness=".7" damping=".05"/>
<geom type="sphere" size=".05" mass=".1"/><site name="distal" pos=".1 0 0"/>
</body></body></worldbody>
<tendon><spatial name="internal"><site site="origin"/><site site="distal"/></spatial></tendon>
</mujoco>'''


def models():
    return mj.MjModel.from_xml_string(XML), mj.MjModel.from_xml_string(derive_xml(XML))


def test_only_declared_compiled_arrays_change():
    a, b = models(); report = audit_models(a, b)
    assert report['passed']
    assert report['changed_arrays'] == ['dof_damping', 'jnt_stiffness']
    np.testing.assert_array_equal(a.site_pos, b.site_pos)
    np.testing.assert_array_equal(a.body_mass, b.body_mass)
    assert a.jnt_stiffness[0] == .4 and b.jnt_stiffness[0] == 0


def test_analytic_translation_and_velocity_controls():
    a, b = models()
    ra, aa = analytic_control(a, a.qpos0)
    rb, ab = analytic_control(b, b.qpos0)
    assert ra['passed'] and rb['passed']
    assert ra['no_active_contacts'] and rb['no_active_contacts']
    assert np.any(aa['velocity_spring'][:3]) and np.any(aa['velocity_damper'][:6])
    np.testing.assert_array_equal(ab['velocity_spring'][:6], np.zeros(6))
    np.testing.assert_array_equal(ab['velocity_damper'][:6], np.zeros(6))
    np.testing.assert_allclose(aa['qfrc_bias'], ab['qfrc_bias'], rtol=0, atol=1e-12)


@pytest.mark.parametrize('name,index', [('body_mass', 1), ('dof_armature', 0),
    ('dof_frictionloss', 0), ('jnt_stiffness', 1), ('tendon_stiffness', 0)])
def test_unrelated_parameter_mutation_rejected(name, index):
    a, b = models(); getattr(b, name)[index] += .01
    with pytest.raises(ValueError, match='Unexpected compiled model change'):
        audit_models(a, b)


def test_solver_change_rejected():
    a, b = models(); b.opt.timestep *= 2
    with pytest.raises(ValueError, match='solver option'):
        audit_models(a, b)


def test_already_unanchored_control_is_not_legacy():
    _, b = models()
    with pytest.raises(ValueError, match='Legacy root stiffness'):
        audit_models(b, b)


def test_missing_intervention_rejected():
    a, _ = models()
    with pytest.raises(ValueError, match='Unexpected compiled model change'):
        audit_models(a, a)
