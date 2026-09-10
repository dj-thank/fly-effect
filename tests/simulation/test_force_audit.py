"""Analytic mechanics controls, not fly biology or trained neural data."""
import numpy as np
import pytest

mj = pytest.importorskip('mujoco')
from organism_core.force_audit import audit_static_forces, PARAMETERS

XML = '''<mujoco><worldbody><body name="fly" pos="0 0 1">
<freejoint name="root"/><geom type="sphere" size=".1" mass="2"/>
</body></worldbody></mujoco>'''


def model():
    return mj.MjModel.from_xml_string(XML)


def test_free_fall_is_accounted_for_but_never_called_standing():
    m = model(); report, arrays = audit_static_forces(m, m.qpos0)
    assert report['passed'] and all(report['checks'].values())
    assert not report['standing_claimed'] and not report['walking_claimed']
    assert not report['root_has_anchor_parameters']
    np.testing.assert_allclose(arrays['qfrc_bias'][:3], [0, 0, 19.62], atol=1e-12)
    np.testing.assert_allclose(arrays['qacc'][:3], [0, 0, -9.81], atol=1e-12)
    np.testing.assert_allclose(arrays['inertial_force'] + arrays['qfrc_bias'], 0, atol=1e-12)


def test_artificial_root_spring_is_exposed_not_silently_removed():
    m = model(); m.jnt_stiffness[0] = 4.
    q = m.qpos0.copy(); q[:3] += [.2, 0, .1]
    report, arrays = audit_static_forces(m, q)
    assert report['passed'] and report['root_has_anchor_parameters']
    assert report['root_passive_parameters']['stiffness'] == 4.
    np.testing.assert_allclose(arrays['qfrc_spring'][:3], [-.8, 0, -.4], atol=1e-12)
    np.testing.assert_allclose(arrays['support_target'][:3], [.8, 0, 20.02], atol=1e-12)
    assert m.jnt_stiffness[0] == 4.


def test_root_damping_is_disclosed_even_when_static_damping_force_is_zero():
    m = model(); m.dof_damping[:] = .5; m.dof_armature[:] = .01
    report, arrays = audit_static_forces(m, m.qpos0)
    assert report['root_has_anchor_parameters'] and report['passed']
    assert report['root_passive_parameters']['armature'] == [.01]*6
    np.testing.assert_array_equal(arrays['qfrc_damper'], np.zeros(6))


def test_gravity_projection_at_rotated_translated_pose():
    m = mj.MjModel.from_xml_string(XML.replace('size=".1"', 'size=".1" pos=".2 .1 0"'))
    q = m.qpos0.copy(); q[:3] += [1, 2, 3]; q[3:7] = [.5, .5, .5, .5]
    report, arrays = audit_static_forces(m, q)
    assert report['passed']
    np.testing.assert_allclose(arrays['qfrc_bias'][:3], [0, 0, 19.62], atol=1e-12)
    assert np.max(np.abs(arrays['qfrc_bias'][3:])) > 0


def test_gravity_disable_flag_is_honored_without_rewriting_model():
    m = model(); m.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_GRAVITY)
    report, arrays = audit_static_forces(m, m.qpos0)
    assert report['gravity_disabled'] and report['passed']
    np.testing.assert_array_equal(arrays['qfrc_bias'], np.zeros(6))
    np.testing.assert_array_equal(m.opt.gravity, [0, 0, -9.81])


def test_internal_joint_spring_and_damping_parameters_are_retained():
    child = '''<body name="leg" pos=".2 0 0"><joint name="hinge" type="hinge"
        stiffness="3" damping=".2"/><geom type="sphere" size=".05" mass=".1" pos=".1 0 0"/></body>'''
    m = mj.MjModel.from_xml_string(XML.replace('</body>', child+'</body>'))
    q = m.qpos0.copy(); q[-1] = .2
    report, arrays = audit_static_forces(m, q)
    assert report['passed'] and not report['root_has_anchor_parameters']
    assert arrays['qfrc_spring'][-1] == pytest.approx(-.6)
    np.testing.assert_array_equal(arrays['model_jnt_stiffness'], [0, 3])


def test_source_qpos_data_and_parameters_are_unchanged():
    m = model(); d = mj.MjData(m); d.qvel[:] = .3; d.qfrc_applied[:] = .7
    before = {n: np.asarray(getattr(m, n)).copy() for n in PARAMETERS}
    q = d.qpos.copy(); v = d.qvel.copy(); f = d.qfrc_applied.copy()
    report, _ = audit_static_forces(m, d.qpos)
    assert report['model_parameters_unchanged'] and report['fresh_zero_input_data']
    np.testing.assert_array_equal(d.qpos, q)
    np.testing.assert_array_equal(d.qvel, v)
    np.testing.assert_array_equal(d.qfrc_applied, f)
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(m, name), values)


@pytest.mark.parametrize('q', [np.zeros(6), np.full(7, np.nan), np.zeros(7)])
def test_invalid_position_is_rejected(q):
    with pytest.raises(ValueError): audit_static_forces(model(), q)


def test_multiple_free_roots_are_not_mislabelled_single_organism():
    extra = '<body pos="1 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>'
    m = mj.MjModel.from_xml_string(XML.replace('</worldbody>', extra+'</worldbody>'))
    with pytest.raises(ValueError, match='one rigid free root'):
        audit_static_forces(m, m.qpos0)


def test_global_passive_callback_is_rejected_and_restored():
    m = model(); previous = mj.get_mjcb_passive()
    try:
        mj.set_mjcb_passive(lambda m, d: None)
        with pytest.raises(ValueError, match='callbacks'): audit_static_forces(m, m.qpos0)
    finally:
        mj.set_mjcb_passive(previous)


def test_report_and_arrays_are_serializable(tmp_path):
    import json
    report, arrays = audit_static_forces(model(), model().qpos0)
    json.dumps(report, allow_nan=False)
    np.savez_compressed(tmp_path/'audit.npz', **arrays)
    with np.load(tmp_path/'audit.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['support_target'], arrays['support_target'])


def test_gravity_compensation_is_not_mislabelled_as_contact_support():
    m = model(); m.body_gravcomp[1] = .25
    report, arrays = audit_static_forces(m, m.qpos0)
    assert report['passed']
    assert arrays['qfrc_gravcomp'][2] == pytest.approx(4.905)
    assert arrays['support_target'][2] == pytest.approx(14.715)
    np.testing.assert_array_equal(arrays['qfrc_constraint'], np.zeros(6))


def test_passive_world_tendon_is_visible_in_budget_and_parameter_snapshot():
    xml = XML.replace('<worldbody>', '<worldbody><site name="world" pos="0 0 0"/>')
    xml = xml.replace('<freejoint name="root"/>', '<freejoint name="root"/><site name="fly_site"/>')
    xml = xml.replace('</mujoco>', '''<tendon><spatial stiffness="2" springlength=".5">
        <site site="world"/><site site="fly_site"/></spatial></tendon></mujoco>''')
    m = mj.MjModel.from_xml_string(xml)
    report, arrays = audit_static_forces(m, m.qpos0)
    assert report['passed']
    assert arrays['qfrc_spring'][2] == pytest.approx(-1.)
    np.testing.assert_array_equal(arrays['model_tendon_stiffness'], [2.])
